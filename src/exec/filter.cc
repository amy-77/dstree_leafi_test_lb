//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"

#include <cmath>

#include <spdlog/spdlog.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "str.h"
#include "scheduler.h"

namespace dstree = upcite::dstree;

class TORCH_API SeriesDataset
    : public torch::data::datasets::Dataset<SeriesDataset> {
 public:
  explicit SeriesDataset(torch::Tensor &series,
                         std::vector<VALUE_TYPE> &targets,
                         int num_instances,
                         torch::Device device) :
      series_(std::move(series.clone())),
      targets_(torch::from_blob(targets.data(),
                                num_instances,
                                torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(device)) {}

  explicit SeriesDataset(torch::Tensor &series,
                         torch::Tensor &targets,
                         int num_instances,
                         torch::Device device) :
      series_(std::move(series.clone())),
      targets_(std::move(targets.clone())) {}

  torch::data::Example<> get(size_t index) override {
    return {series_[index], targets_[index]};
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

  bsf_distances_.reserve(config.filter_train_nexample_);
  nn_distances_.reserve(config.filter_train_nexample_);
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
//#ifndef DEBUGGED
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
//#ifndef DEBUGGED
    bool is_train_logged = false;
//#endif
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
      model_->eval();
      torch::NoGradGuard no_grad;

      torch::Tensor prediction = model_->forward(valid_data);

//      valid_loss = huber_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();
      valid_loss = mse_loss->forward(prediction, valid_target).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);

//#ifndef DEBUGGED
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
//#endif
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
//#ifndef DEBUGGED
    if ((epoch < 100 && epoch % 10 == 0) || (epoch >= 100 && epoch % 100 == 0)) {
      model_->eval();
      torch::NoGradGuard no_grad;
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
//#endif
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
    model_->eval();
    torch::NoGradGuard no_grad;

    auto prediction = model_->forward(all_data).detach().cpu();

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

    if (config_.get().filter_remove_square_) {
      return pred * pred;
    } else {
      return pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}
