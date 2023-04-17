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
#include "comp.h"
#include "interval.h"
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
    return {series_[index], targets_[index]};
  }

  torch::optional<size_t> size() const override {
    return targets_.size(0);
  }

 private:
  torch::Tensor series_, targets_;
};

dstree::Filter::Filter(dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    config_(config),
    id_(id),
    is_active_(false),
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

  // delayed until allocated
  model_ = nullptr;

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

  if (is_active_ && model_ == nullptr) {
    // TODO instantiate the model according to the assigned model_setting_
    model_ = std::make_unique<dstree::MLPFilter>(config_.get().series_length_,
                                                 config_.get().filter_dim_latent_,
                                                 config_.get().filter_train_dropout_p_,
                                                 config_.get().filter_leaky_relu_negative_slope_);
    model_->to(*device_);
  }

  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

  if (config_.get().filter_remove_square_) {
    for (ID_TYPE i = 0; i < train_size_; ++i) {
      nn_distances_[i] = sqrt(nn_distances_[i]);
      bsf_distances_[i] = sqrt(bsf_distances_[i]);
    }

    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < train_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }
  }

#ifdef DEBUG
  if (!lb_distances_.empty()) {
    spdlog::debug("filter {:d} stream {:d} d_node{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(lb_distances_.data(), train_size_));
  }

  spdlog::debug("filter {:d} stream {:d} d_bsf{:s} = {:s}",
                id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                upcite::array2str(bsf_distances_.data(), train_size_));

  spdlog::debug("filter {:d} stream {:d} d_nn{:s} = {:s}",
                id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                upcite::array2str(nn_distances_.data(), train_size_));
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

  torch::optim::SGD optimizer(model_->parameters(), config_.get().filter_train_learning_rate_);
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(optimizer);

//  torch::nn::HuberLoss huber_loss(torch::nn::HuberLossOptions().reduction(torch::kMean).delta(1.0));
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);

  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif

  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
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
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0)
                               / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
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
#endif
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} stream {:d} t_losses = {:s}",
                id_, stream_id, upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} stream {:d} v_losses = {:s}",
                id_, stream_id, upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

//    torch::NoGradGuard no_grad;
    c10::InferenceMode guard;
    model_->eval();

    auto prediction = model_->forward(shared_train_queries_).detach().cpu();
    VALUE_TYPE *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();

    pred_distances_.insert(pred_distances_.end(), predictions_array, predictions_array + train_size_);

#ifdef DEBUG
    spdlog::info("filter {:d} stream {:d} d_pred{:s} = {:s}",
                 id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                 upcite::array2str(predictions_array, train_size_));
#endif

    if (config_.get().filter_is_conformal_) {
      ID_TYPE num_conformal_examples = num_valid_examples * config_.get().filter_conformal_train_val_split_;
      auto residuals = upcite::make_reserved<VALUE_TYPE>(num_conformal_examples);

      for (ID_TYPE i = 0; i < num_conformal_examples; ++i) {
        // TODO torch::Tensor to ptr is not stable
        if (predictions_array[num_train_examples + i] > constant::MIN_VALUE &&
            predictions_array[num_train_examples + i] < constant::MAX_VALUE &&
            !upcite::equals_zero(predictions_array[num_train_examples + i])) {
          // TODO not necessary for global symmetrical confidence intervals
          residuals.emplace_back(predictions_array[num_train_examples + i] - nn_distances_[num_train_examples + i]);
        }
      }

      conformal_predictor_->fit(residuals);

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::info(
          "filter {:d} stream {:d} conformal confidence (one side-)interval {:.3f}@0.50, {:.3f}@0.90, {:.3f}@0.95, {:.3f}@0.99",
          stream_id,
          id_,
          conformal_predictor_->get_alpha(0.5),
          conformal_predictor_->get_alpha(0.9),
          conformal_predictor_->get_alpha(0.95),
          conformal_predictor_->get_alpha(0.99));

      if (!upcite::is_equal(config_.get().filter_conformal_confidence_, static_cast<VALUE_TYPE>(0.5)) &&
          !upcite::is_equal(config_.get().filter_conformal_confidence_, static_cast<VALUE_TYPE>(0.9)) &&
          !upcite::is_equal(config_.get().filter_conformal_confidence_, static_cast<VALUE_TYPE>(0.95)) &&
          !upcite::is_equal(config_.get().filter_conformal_confidence_, static_cast<VALUE_TYPE>(0.99))) {
        spdlog::info("filter {:d} stream {:d} conformal confidence (one side-)interval {:.3f}@{:.3f}",
                     stream_id,
                     id_,
                     conformal_predictor_->get_alpha(config_.get().filter_conformal_confidence_),
                     config_.get().filter_conformal_confidence_);
      }
#endif
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

    spdlog::info("filter {:d} gpu mem = {:.3f}MB", id_, (static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024)));
  }
#endif

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

  // TODO condense size indicators into a bitmap, as they all = train_size_
  // bsf_distances_
  ID_TYPE size_placeholder = bsf_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!bsf_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(bsf_distances_.data()), sizeof(VALUE_TYPE) * bsf_distances_.size());
  }

  // nn_distances_
  size_placeholder = nn_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!nn_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(nn_distances_.data()), sizeof(VALUE_TYPE) * nn_distances_.size());
  }

  // lb_distances_
  size_placeholder = lb_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!lb_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(lb_distances_.data()), sizeof(VALUE_TYPE) * lb_distances_.size());
  }

  // ub_distances_
  size_placeholder = ub_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!ub_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(ub_distances_.data()), sizeof(VALUE_TYPE) * ub_distances_.size());
  }

  // pred_distances_
  size_placeholder = pred_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!pred_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(pred_distances_.data()), sizeof(VALUE_TYPE) * pred_distances_.size());
  }

  if (is_active_) {
    size_placeholder = model_setting_.model_setting_str.size();
  } else {
    size_placeholder = 0;
  }
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (is_active_) {
    node_fos.write(reinterpret_cast<const char *>(model_setting_.model_setting_str.data()),
                   sizeof(model_setting_.model_setting_str));
  }

  ID_TYPE is_trained_placeholder = 0;
  if (is_trained_) {
    is_trained_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (is_trained_) {
    std::string model_filepath = config_.get().dump_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;

    torch::save(model_, model_filepath);
  }

  ID_TYPE is_conformal_placeholder = 0;
  if (config_.get().filter_is_conformal_) {
    is_conformal_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_conformal_placeholder), sizeof(ID_TYPE));
  if (config_.get().filter_is_conformal_) {
    conformal_predictor_->dump(node_fos);
  }

  return SUCCESS;
}

RESPONSE dstree::Filter::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  // train_size_
  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  train_size_ = ifs_id_buf[0];

  // bsf_distances_
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    bsf_distances_.insert(bsf_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // nn_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    nn_distances_.insert(nn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // lb_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    lb_distances_.insert(lb_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // ub_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    ub_distances_.insert(ub_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // pred_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    pred_distances_.insert(pred_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // model_setting_
  is_active_ = false;

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    std::string model_setting_str;
    model_setting_str.resize(size_indicator);
    node_ifs.read(const_cast<char *>(model_setting_str.data()), size_indicator);

    if (config_.get().to_load_filters_) {
      model_setting_ = MODEL_SETTING(model_setting_str);

      is_active_ = true;
    }
  }

  // model_
  is_trained_ = false;

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0 && config_.get().to_load_filters_) {
    std::string model_filepath = config_.get().load_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;

    if (!fs::is_regular_file(model_filepath)) {
      spdlog::error("Empty model_filepath found: {:s}", model_filepath);
      return FAILURE;
    }

    // TODO instantiate the model according to the setting
    model_ = std::make_unique<dstree::MLPFilter>(config_.get().series_length_,
                                                 config_.get().filter_dim_latent_,
                                                 config_.get().filter_train_dropout_p_,
                                                 config_.get().filter_leaky_relu_negative_slope_);
    model_->to(*device_);

    torch::load(model_, model_filepath);

    model_->eval();
//  net->to(torch::Device(torch::kCPU));
    c10::cuda::CUDACachingAllocator::emptyCache();

    is_trained_ = true;
  }

  // conformal_predictor_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    // TODO load only if (config_.get().to_load_filters_); the current implementation works though
    conformal_predictor_->load(node_ifs, ifs_buf);
  }

  return SUCCESS;
}

VALUE_TYPE dstree::Filter::get_external_pruning_frequency() const {
  if (lb_distances_.empty() || lb_distances_.size() != bsf_distances_.size()) {
    return 0;
  }

  ID_TYPE pruned_counter = 0;
  for (ID_TYPE i = 0; i < lb_distances_.size(); ++i) {
    if (lb_distances_[i] > bsf_distances_[i]) {
      pruned_counter += 1;
    }
  }

  return static_cast<VALUE_TYPE>(pruned_counter) / static_cast<VALUE_TYPE>(lb_distances_.size());
}
