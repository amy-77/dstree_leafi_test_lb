//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"

#include <boost/format.hpp>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "str.h"

namespace dstree = upcite::dstree;

class TORCH_API SeriesDataset : public torch::data::datasets::Dataset<SeriesDataset> {
 public:
  explicit SeriesDataset(torch::Tensor &series,
                         std::vector<VALUE_TYPE> &targets,
                         int num_instances,
                         torch::Device device) :
      series_(series),
      targets_(torch::from_blob(targets.data(),
                                num_instances,
                                torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(device)) {}

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

dstree::Filter::Filter(upcite::Logger &logger,
                       dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    logger_(logger),
    config_(config),
    id_(id),
    shared_train_queries_(shared_train_queries),
    is_trained_(false),
    train_size_(0) {
//  torch::Device device = torch::Device(c10::DeviceType::Lazy);

  if (config.nf_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config.nf_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  model_ = std::make_unique<dstree::MLP>(config.series_length_,
                                         config.nf_dim_latent_,
                                         config.nf_train_dropout_p_,
                                         config.nf_leaky_relu_negative_slope_);
  model_->to(*device_);

  bsf_distances_.reserve(config.nf_train_nexample_);
  nn_distances_.reserve(config.nf_train_nexample_);
}

RESPONSE dstree::Filter::train() {
  if (is_trained_) {
    return FAILURE;
  }

#ifdef DEBUG
//#ifndef DEBUGGED
  if (train_size_ < config_.get().nf_train_nexample_) {
    MALAT_LOG(logger_.get().logger, trivial::error) << boost::format(
          "%d train examples collected; expected %d")
          % train_size_
          % config_.get().nf_train_nexample_;
  }
//#endif
#endif

  ID_TYPE stream_id = -1;
  if (config_.get().nf_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().nf_device_id_).id(); // compiles with libtorch-gpu
  }

#ifdef DEBUG
  if (!node_lower_bound_distances_.empty()) {
    MALAT_LOG(logger_.get().logger, trivial::debug) << boost::format(
          "stream %d filter %d node distances = %s")
          % stream_id
          % id_
          % upcite::get_str(node_lower_bound_distances_.data(), train_size_);
  }

  MALAT_LOG(logger_.get().logger, trivial::debug) << boost::format(
        "stream %d filter %d bsf distances = %s")
        % stream_id
        % id_
        % upcite::get_str(bsf_distances_.data(), train_size_);

  MALAT_LOG(logger_.get().logger, trivial::debug) << boost::format(
        "stream %d filter %d nn distances = %s")
        % stream_id
        % id_
        % upcite::get_str(nn_distances_.data(), train_size_);

#endif

  ID_TYPE num_train_examples = train_size_ / 6 * 5;
//    int num_valid_examples = train_size_ - num_train_examples;

  auto dataset = SeriesDataset(shared_train_queries_, nn_distances_, num_train_examples, *device_);
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      dataset.map(torch::data::transforms::Stack<>()), config_.get().nf_train_batchsize_);

  torch::optim::SGD optimizer(model_->parameters(), config_.get().nf_train_learning_rate_);

#ifdef DEBUG
  std::vector<float> global_losses, local_losses;
  global_losses.reserve(config_.get().nf_train_nepoch_);
  local_losses.reserve(num_train_examples / config_.get().nf_train_batchsize_ + 1);
#endif

  for (size_t epoch = 0; epoch < config_.get().nf_train_nepoch_; ++epoch) {
    adjust_learning_rate(optimizer,
                         config_.get().nf_train_learning_rate_,
                         config_.get().nf_train_min_lr_,
                         epoch,
                         config_.get().nf_train_nepoch_);

    for (auto &batch : *data_loader) {
      auto batch_data = batch.data, batch_target = batch.target;

      optimizer.zero_grad();

      torch::Tensor prediction = model_->forward(batch_data);
      torch::Tensor loss = torch::mse_loss(prediction, batch_target);

      loss.backward();
      auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                    config_.get().nf_train_clip_grad_max_norm_,
                                                    config_.get().nf_train_clip_grad_norm_type_);
      optimizer.step();

#ifdef DEBUG
      local_losses.emplace_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    global_losses.emplace_back(std::accumulate(local_losses.begin(), local_losses.end(), 0.0)
                                   / (float) local_losses.size());
    local_losses.clear();
#endif
  }

#ifdef DEBUG
  MALAT_LOG(logger_.get().logger, trivial::info) << boost::format(
        "stream %d filter %d losses = %s")
        % stream_id
        % id_
        % upcite::get_str(global_losses.data(), config_.get().nf_train_nepoch_);
#endif
  optimizer.zero_grad();
  for (const auto &parameter : model_->parameters()) {
    parameter.requires_grad_(false);
  }

//    net->to(torch::Device(torch::kCPU));
  model_->eval();

#ifdef DEBUG
  {
    torch::NoGradGuard no_grad;

    auto predictions = model_->forward(shared_train_queries_).detach().to(torch::Device(torch::kCPU));

    MALAT_LOG(logger_.get().logger, trivial::info) << boost::format(
          "stream %d filter %d predictions = %s")
          % stream_id
          % id_
          % upcite::get_str(predictions.accessor<VALUE_TYPE, 1>().data(), train_size_);
  };
#endif

#ifdef DEBUG
  if (torch::cuda::is_available()) {
    size_t memory_size = 0;

    for (const auto &parameter : model_->parameters()) {
      memory_size += parameter.nbytes();
    }

    for (const auto &buffer : model_->buffers()) {
      memory_size += buffer.nbytes();
    }

    if (id_ == 0) {
      MALAT_LOG(logger_.get().logger, trivial::info) << boost::format(
            "filter %d size = %.3fMB")
            % id_
            % (static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024));
    }
  }
#endif

  c10::cuda::CUDACachingAllocator::emptyCache();
  is_trained_ = true;

  return SUCCESS;
}

VALUE_TYPE dstree::Filter::infer(torch::Tensor &query_series) const {
  if (is_trained_) {
    torch::NoGradGuard no_grad;

    return model_->infer(query_series).item<VALUE_TYPE>();
  } else {
    return constant::MAX_VALUE;
  }
}
