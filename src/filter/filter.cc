//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"

#include <cmath>

#include <boost/filesystem.hpp>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "spdlog/spdlog.h"

#include "str.h"
#include "vec.h"
#include "intervel.h"
#include "scheduler.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

class TORCH_API SeriesDataset
    : public torch::data::datasets::Dataset<SeriesDataset> {
 public:
  SeriesDataset(torch::Tensor &series,
                std::vector<VALUE_TYPE> &targets,
                int num_instances,
                torch::Device device) :
      series_(std::move(series.clone())),
      targets_(torch::from_blob(targets.data(),
                                num_instances,
                                torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(device)) {
  }

  SeriesDataset(torch::Tensor &series,
                torch::Tensor &targets,
                int num_instances,
                torch::Device device) :
      series_(std::move(series.clone())),
      targets_(std::move(targets.clone())) {
  }

  torch::data::Example<> get(size_t index) override {
    return {
        series_[index], targets_[index]};
  }

  torch::optional<size_t> size() const override {
    return targets_.size(0);
  }

 private:
  torch::Tensor series_, targets_;
};

void adjust_learning_rate(torch::optim::SGD &optimizer,
                          VALUE_TYPE max_lr,
                          VALUE_TYPE min_lr,
                          ID_TYPE epoch,
                          ID_TYPE max_epoch) {
//    float new_lr = max_lr - (max_lr - min_lr) * ((float) epoch / (float) max_epoch);

  float boundary = 9;
  float current_step = (epoch / max_epoch - 0.5f) * boundary;
  float new_lr = min_lr + (max_lr - min_lr) / (1 + exp(current_step));

  for (auto &group : optimizer.param_groups()) {
    if (group.has_options()) {
      group.options().set_lr(new_lr);
    }
  }
}

dstree::Filter::Filter(dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    config_(config),
    id_(id),
    shared_train_queries_(shared_train_queries),
    is_trained_(false),
    train_size_(0) {
//  torch::Device device = torch::Device(c10::DeviceType::Lazy);

  if (config.filter_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config.filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  model_ = std::make_unique<dstree::MLP>(config.series_length_,
                                         config.filter_dim_latent_,
                                         config.filter_train_dropout_p_,
                                         config.filter_leaky_relu_negative_slope_);
  model_->to(*device_);

  if (!config.to_load_index_) {
    bsf_distances_.reserve(config.filter_train_nexample_);
    nn_distances_.reserve(config.filter_train_nexample_);
  }

  if (config.filter_is_conformal_) {
    conformal_predictor_ = std::make_unique<upcite::ConformalRegressor>(config.filter_conformal_core_type_,
                                                                        config.filter_conformal_confidence_);
  } else {
    conformal_predictor_ = nullptr;
  }
}

RESPONSE dstree::Filter::train() {
  if (is_trained_) {
    return FAILURE;
  }

#ifdef DEBUG
//#ifndef DEBUGGED
  if (train_size_ < config_.get().filter_train_nexample_) {
    spdlog::error("{:d} train examples collected; expected {:d}",
                  train_size_, config_.get().filter_train_nexample_);

    // TODO bind a wrapper for spdlog::shutdown and exit
    spdlog::shutdown();
    exit(FAILURE);
  }
//#endif
#endif

  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

#ifdef DEBUG
  if (!node_lower_bound_distances_.empty()) {
    spdlog::debug("stream {:d} filter {:d} d_node_sq = {:s}",
                  stream_id, id_, upcite::get_str(node_lower_bound_distances_.data(), train_size_));
  }

  spdlog::debug("stream {:d} filter {:d} d_bsf_sq = {:s}",
                stream_id, id_, upcite::get_str(bsf_distances_.data(), train_size_));
#endif

  if (config_.get().filter_remove_square_) {
    for (ID_TYPE i = 0; i < train_size_; ++i) {
      nn_distances_[i] = sqrt(nn_distances_[i]);
    }
  }

#ifdef DEBUG
  spdlog::debug("stream {:d} filter {:d} d_nn{:s} = {:s}",
                stream_id, id_, config_.get().filter_remove_square_ ? "" : "_sq",
                upcite::get_str(nn_distances_.data(), train_size_));
#endif

  ID_TYPE num_train_examples = train_size_ * config_.get().filter_train_val_split_;

  auto train_data = shared_train_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
  auto train_dataset = SeriesDataset(train_data, nn_distances_, num_train_examples, *device_);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

  ID_TYPE num_valid_examples = train_size_ - num_train_examples;
  auto valid_data = shared_train_queries_.get().index({torch::indexing::Slice(
      num_train_examples, train_size_)}).clone();
  auto valid_target = torch::from_blob(nn_distances_.data() + num_train_examples,
                                       num_valid_examples,
                                       torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  auto valid_dataset = SeriesDataset(valid_data, valid_target, num_valid_examples, *device_);
  auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      valid_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

#ifdef DEBUG
#ifndef DEBUGGED
  auto all_data = shared_train_queries_.get().clone();
  auto all_dataset = SeriesDataset(all_data, nn_distances_, train_size_, *device_);
  auto all_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      all_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

#ifndef DEBUGGED
  spdlog::info("stream {:d} filter {:d} v shape = {:s}; {:s}",
               stream_id, id_,
               upcite::get_str(valid_data.sizes().data(), valid_data.sizes().size()),
               upcite::get_str(valid_target.sizes().data(), valid_target.sizes().size()));
#endif
  auto last_query_cpu = train_data.index({num_train_examples - 1}).cpu().contiguous();
  spdlog::info("stream {:d} filter {:d} tq[-1] {:b} = {:s}",
               stream_id, id_, last_query_cpu.requires_grad(),
               upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));

  last_query_cpu = valid_data.index({num_valid_examples - 1}).cpu().contiguous();
  spdlog::info("stream {:d} filter {:d} vq[-1] {:b} = {:s}",
               stream_id, id_, last_query_cpu.requires_grad(),
               upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));

  last_query_cpu = all_data.index({train_size_ - 1}).cpu().contiguous();
  spdlog::info("stream {:d} filter {:d} eq[-1] {:b} = {:s}",
               stream_id, id_, last_query_cpu.requires_grad(),
               upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));
#endif
#endif

  torch::optim::SGD optimizer(model_->parameters(), config_.get().filter_train_learning_rate_);
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(optimizer);

//  torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().reduction(torch::kMean).delta(1.0));
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

#ifdef DEBUG
//#ifndef DEBUGGED
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);

  valid_losses.reserve(config_.get().filter_train_nepoch_);
//#endif
#endif

  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
#ifdef DEBUG
#ifndef DEBUGGED
    bool is_train_logged = false;
#endif
#endif

    model_->train();

    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;

      optimizer.zero_grad();

      torch::Tensor prediction = model_->forward(batch_data);

//      torch::Tensor loss = huber_loss->forward(prediction, batch_target);
      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      loss.backward();

      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer.step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());

#ifndef DEBUGGED
      if (!is_train_logged && ((epoch < 100 && epoch % 10 == 0) || (epoch >= 100 && epoch % 100 == 0))) {
        auto target_cpu = batch_target.detach().cpu().contiguous();
        spdlog::info("stream {:d} filter {:d} t{:d} b0 d_nn{:s} = {:s}",
                     stream_id, id_, epoch, config_.get().filter_remove_square_ ? "" : "_sq",
                     upcite::get_str(target_cpu.data_ptr<VALUE_TYPE>(), target_cpu.size(0)));

        auto pred_cpu = prediction.detach().cpu().contiguous();
        spdlog::info("stream {:d} filter {:d} t{:d} b0 d_pred{:s} = {:s}",
                     stream_id, id_, epoch, config_.get().filter_remove_square_ ? "" : "_sq",
                     upcite::get_str(pred_cpu.data_ptr<VALUE_TYPE>(), pred_cpu.size(0)));

        auto last_query_cpu = batch_data.index({batch_data.size(0) - 1}).cpu().contiguous();
        spdlog::info("stream {:d} filter {:d} t{:d} b0 q[-1] {:b} = {:s}",
                     stream_id, id_, epoch, batch_data.requires_grad(),
                     upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));

#ifndef DEBUGGED
        spdlog::info("stream {:d} filter {:d} t{:d} b0 shape = {:s}; {:s}",
                     stream_id, id_, epoch,
                     upcite::get_str(batch_data.sizes().data(), batch_data.sizes().size()),
                     upcite::get_str(batch_target.sizes().data(), batch_target.sizes().size()));

        spdlog::info("stream {:d} filter {:d} t{:d} = {:b}, {:b}, {:b}",
                     stream_id, id_, epoch,
                     model_->fc1_->is_training(), model_->fc3_->is_training(), model_->activate_->is_training());
#endif

        is_train_logged = true;
      }
#endif
#endif
    }

#ifdef DEBUG
//#ifndef DEBUGGED
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0)
                               / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
//#endif
#endif

    VALUE_TYPE valid_loss = 0;

    { // evaluate
//      torch::NoGradGuard no_grad;
      c10::InferenceMode guard;
      model_->eval();

      torch::Tensor prediction = model_->forward(valid_data);

//      valid_loss = huber_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();
      valid_loss = mse_loss->forward(prediction, valid_target).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);

#ifndef DEBUGGED
      if ((epoch < 100 && epoch % 10 == 0) || (epoch >= 100 && epoch % 100 == 0)) {
#ifndef DEBUGGED
        spdlog::info("stream {:d} filter {:d} v{:d} = {:b}, {:b}, {:b}",
                     stream_id, id_, epoch,
                     model_->fc1_->is_training(), model_->fc3_->is_training(), model_->activate_->is_training());
#endif

        auto pred_cpu = prediction.detach().cpu().contiguous();
        spdlog::info("stream {:d} filter {:d} v{:d} d_pred{:s} = {:s}",
                     stream_id, id_, epoch, config_.get().filter_remove_square_ ? "" : "_sq",
                     upcite::get_str(pred_cpu.data_ptr<VALUE_TYPE>(), pred_cpu.size(0)));

        auto last_query_cpu = valid_data.index({num_valid_examples - 1}).cpu().contiguous();
        spdlog::info("stream {:d} filter {:d} v{:d} q[-1] {:b} = {:s}",
                     stream_id, id_, epoch, valid_data.requires_grad(),
                     upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));
      }
#endif
#endif
    }

    lr_scheduler.check_step(valid_loss);
    // TODO refactor to SigmoidLR
//    adjust_learning_rate(optimizer,
//                         config_.get().filter_train_learning_rate_,
//                         config_.get().filter_train_min_lr_,
//                         epoch,
//                         config_.get().filter_train_nepoch_);

#ifdef DEBUG
#ifndef DEBUGGED
    if ((epoch < 100 && epoch % 10 == 0) || (epoch >= 100 && epoch % 100 == 0)) {
//      torch::NoGradGuard no_grad;
      c10::InferenceMode guard;
      model_->eval();

      torch::Tensor pred_cpu;

//      pred_cpu = model_->forward(all_data).detach().cpu().contiguous();

      for (auto &batch : *all_data_loader) { // only one batch
        batch_data = batch.data;
        pred_cpu = model_->forward(batch_data).detach().cpu().contiguous();
      }

      spdlog::info("stream {:d} filter {:d} e{:d} d_pred{:s} = {:s}",
                   stream_id, id_, epoch, config_.get().filter_remove_square_ ? "" : "_sq",
                   upcite::get_str(pred_cpu.data_ptr<VALUE_TYPE>(), pred_cpu.size(0)));

      auto last_query_cpu = batch_data.index({train_size_ - 1}).cpu().contiguous();
      spdlog::info("stream {:d} filter {:d} e{:d} q[-1] {:b} = {:s}",
                   stream_id, id_, epoch, batch_data.requires_grad(),
                   upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)));

#ifndef DEBUGGED
      spdlog::info("stream {:d} filter {:d} e{:d} = {:b}, {:b}, {:b}",
                   stream_id, id_, epoch,
                   model_->fc1_->is_training(), model_->fc3_->is_training(), model_->activate_->is_training());
#endif
    };
#endif
#endif
  }

#ifdef DEBUG
  spdlog::debug("stream {:d} filter {:d} t_losses = {:s}",
                stream_id, id_, upcite::get_str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("stream {:d} filter {:d} v_losses = {:s}",
                stream_id, id_, upcite::get_str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

#ifdef DEBUG
//#ifndef DEBUGGED
  {
//    torch::NoGradGuard no_grad;
    c10::InferenceMode guard;
    model_->eval();

//    auto prediction = model_->forward(all_data).detach().cpu();
    auto prediction = model_->forward(shared_train_queries_).detach().cpu();

    spdlog::info("stream {:d} filter {:d} d_pred{:s} = {:s}",
                 stream_id, id_, config_.get().filter_remove_square_ ? "" : "_sq",
                 upcite::get_str(prediction.accessor<VALUE_TYPE, 1>().data(), train_size_));

#ifndef DEBUGGED
    spdlog::info("stream {:d} filter {:d} = {:b}, {:b}, {:b}",
                 stream_id, id_,
                 model_->fc1_->is_training(), model_->fc3_->is_training(), model_->activate_->is_training());
#endif
  };
//#endif
#endif

  if (config_.get().filter_is_conformal_) {
    ID_TYPE num_conformal_examples = num_valid_examples * config_.get().filter_conformal_train_val_split_;

    c10::InferenceMode guard;
    model_->eval();

    torch::Tensor prediction = model_->forward(valid_data);
    auto predictions_array = prediction.accessor<VALUE_TYPE, 1>().data();

    std::vector<VALUE_TYPE> residuals;
    residuals.reserve(num_conformal_examples);
    for (ID_TYPE i = 0; i < num_conformal_examples; ++i) {
      residuals.emplace_back(predictions_array[i] - nn_distances_[num_train_examples + i]);
    }

    conformal_predictor_->fit(residuals);

#ifdef DEBUG
//#ifndef DEBUGGED
    auto y_hat = upcite::make_reserved<VALUE_TYPE>(num_valid_examples - num_conformal_examples);
    y_hat.assign(predictions_array + num_conformal_examples, predictions_array + num_valid_examples);

    std::vector<INTERVAL> y_intervals = conformal_predictor_->predict(y_hat);

    spdlog::info("stream {:d} filter {:d} d_interval{:s} eval = {:s}",
                 stream_id, id_, config_.get().filter_remove_square_ ? "" : "_sq",
                 upcite::get_str(y_intervals.data(), y_intervals.size()));
//#endif
#endif
  }

#ifdef DEBUG
  if (torch::cuda::is_available() && id_ == 0) {
    size_t memory_size = 0;

    for (const auto &parameter : model_->parameters()) {
      memory_size += parameter.nbytes();
    }

    for (const auto &buffer : model_->buffers()) {
      memory_size += buffer.nbytes();
    }

    spdlog::info("filter {:d} size = {:.3f}MB", id_, (static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024)));
  }
#endif

//  for (const auto &parameter : model_->parameters()) {
//    parameter.requires_grad_(false);
//  }
  model_->eval();
//  net->to(torch::Device(torch::kCPU));

  c10::cuda::CUDACachingAllocator::emptyCache();
  is_trained_ = true;

  return SUCCESS;
}

VALUE_TYPE dstree::Filter::infer(torch::Tensor &query_series) const {
  if (is_trained_) {
    torch::NoGradGuard no_grad;

    VALUE_TYPE pred = model_->forward(query_series).item<VALUE_TYPE>();
    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
    }

    if (config_.get().filter_remove_square_) {
      return pred * pred;
    } else {
      return pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}

RESPONSE dstree::Filter::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&train_size_), sizeof(ID_TYPE));

  node_fos.write(reinterpret_cast<const char *>(bsf_distances_.data()), sizeof(VALUE_TYPE) * bsf_distances_.size());
  node_fos.write(reinterpret_cast<const char *>(nn_distances_.data()), sizeof(VALUE_TYPE) * nn_distances_.size());

  if (config_.get().filter_is_conformal_) {
    conformal_predictor_->dump(node_fos);
  }

  std::string model_filepath = config_.get().dump_filters_folderpath_ + std::to_string(id_) +
      config_.get().model_dump_file_postfix_;

  torch::save(model_, model_filepath);

  return SUCCESS;
}

RESPONSE dstree::Filter::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  // train_size_
  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  train_size_ = ifs_id_buf[0];

  read_nbytes = sizeof(VALUE_TYPE) * train_size_;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  bsf_distances_.insert(bsf_distances_.begin(), ifs_value_buf, ifs_value_buf + train_size_);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  nn_distances_.insert(nn_distances_.begin(), ifs_value_buf, ifs_value_buf + train_size_);

  if (config_.get().filter_is_conformal_) {
    // TODO support ad hoc cases?
    conformal_predictor_->load(node_ifs, ifs_buf);
  }

  std::string model_filepath = config_.get().load_filters_folderpath_ + std::to_string(id_) +
      config_.get().model_dump_file_postfix_;

  if (!fs::is_regular_file(model_filepath)) {
    spdlog::error("Empty model_filepath found: {:s}", model_filepath);
    return FAILURE;
  }

  torch::load(model_, model_filepath);
  model_->eval();
//  net->to(torch::Device(torch::kCPU));

  c10::cuda::CUDACachingAllocator::emptyCache();
  is_trained_ = true;

  return SUCCESS;
}
