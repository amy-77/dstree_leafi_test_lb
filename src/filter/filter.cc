//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"

#include <cmath>
#include <chrono>
#include <random>
#include <iostream>

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
#include "dataset.h"
#include "scheduler.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Filter::Filter(dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    config_(config),
    id_(id),
    is_active_(false),
    global_queries_(shared_train_queries),
    is_trained_(false),
    is_distances_preprocessed_(false),
    is_distances_logged(false),
    global_data_size_(0),
    local_data_size_(0),
    model_setting_ref_(MODEL_SETTING_PLACEHOLDER_REF) {
  if (config.filter_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                              static_cast<c10::DeviceIndex>(config.filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  // delayed until allocated (either in trial or activation)
  model_ = nullptr;

  if (!config.to_load_index_ && config.filter_train_nexample_ > 0) {
    global_bsf_distances_.reserve(config.filter_train_nexample_);
    global_lnn_distances_.reserve(config.filter_train_nexample_);
    lb_distances_.reserve(config.filter_train_nexample_);
  }

  if (config.filter_is_conformal_) {
    conformal_predictor_ = std::make_unique<upcite::ConformalRegressor>(config.filter_conformal_core_type_,
                                                                        config.filter_conformal_confidence_);
  } else {
    conformal_predictor_ = nullptr;
  }
}

RESPONSE dstree::Filter::fit_conformal_predictor(bool is_trial, bool collect_runtime_stat) {
  ID_TYPE num_conformal_examples;

  if (!collect_runtime_stat) {
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;

    num_conformal_examples = num_global_valid_examples;
  } else {
    if (config_.get().filter_train_num_global_example_ > 0 && config_.get().filter_train_num_local_example_) {
      ID_TYPE num_global_train_examples = config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else if (config_.get().filter_train_nexample_ > 0) {
      ID_TYPE num_global_train_examples = config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_nexample_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else {
      num_conformal_examples = 8192;
    }
  }

  auto residuals = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

  if (collect_runtime_stat) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // include two sentry diffs
    for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
      residuals.push_back(dist(e2));
    }
  } else {
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;

    VALUE_TYPE max_diff = constant::MIN_VALUE, mean_diff = 0, std_diff = 0;
    ID_TYPE num_diff = 0;

    for (ID_TYPE i = 0; i < num_conformal_examples; ++i) {
      // TODO torch::Tensor to ptr is not stable
      if (global_pred_distances_[num_global_train_examples + i] > constant::MIN_VALUE &&
          global_pred_distances_[num_global_train_examples + i] < constant::MAX_VALUE &&
          !upcite::equals_zero(global_pred_distances_[num_global_train_examples + i])) {
        // TODO not necessary for global symmetrical confidence intervals
        VALUE_TYPE diff = abs(global_pred_distances_[num_global_train_examples + i]
                                  - global_lnn_distances_[num_global_train_examples + i]);

        if (diff > max_diff) {
          max_diff = diff;
        }
        mean_diff += diff;
        num_diff += 1;

        residuals.emplace_back(diff);
      }
    }

    if (num_diff < num_conformal_examples) {
      spdlog::error("conformal filter {:d} model {:s} collected {:d} pred diff; expected {:d}",
                    id_, model_setting_ref_.get().model_setting_str, num_diff, num_conformal_examples);
    }

    mean_diff /= num_diff;

    for (ID_TYPE diff_i = 0; diff_i < num_diff; ++diff_i) {
      std_diff += (residuals[diff_i] - mean_diff) * (residuals[diff_i] - mean_diff);
    }

    std_diff = sqrt(std_diff / num_diff);
    VALUE_TYPE max_normal_value = mean_diff + 3 * std_diff + constant::EPSILON_GAP;

    max_diff += constant::EPSILON_GAP;
    if (max_normal_value < max_diff) {
      max_normal_value = max_diff;
    }

    // add the first of two sentries: 0
    residuals.push_back(0);

    // add the second of two sentries: max range upper boundary
    // previously using the max pred value
    residuals.push_back(max_normal_value);
  }

  if (is_trial && !collect_runtime_stat) {
    for (auto &residual : residuals) { residual = residual < 0 ? -residual : residual; }
    std::sort(residuals.begin(), residuals.end());

    auto residual_i = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(residuals.size())
        * config_.get().filter_trial_confidence_level_);

    conformal_predictor_->set_alpha(residuals[residual_i], true);

#ifdef DEBUG
//#ifndef DEBUGGED
    spdlog::debug("trial filter {:d} model {:s} trial error (half-)interval = {:.3f} @ {:.2f}",
                  id_, model_setting_ref_.get().model_setting_str,
                  get_abs_error_interval(),
                  config_.get().filter_trial_confidence_level_);
//#endif
#endif
  } else if (!is_trial && collect_runtime_stat) {
    conformal_predictor_->fit(residuals);

    if (config_.get().filter_conformal_is_smoothen_) {
      auto recalls = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 1);

      for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
        recalls.push_back(dist(e2));
      }

      std::sort(recalls.begin(), recalls.end()); //non-decreasing

      fit_filter_conformal_spline(recalls);
    }
  } else if (!is_trial && !collect_runtime_stat) {
    conformal_predictor_->fit(residuals);
  } else {
    spdlog::error("trial filter {:d} model {:s} both trial and collect modes were triggered",
                  id_, model_setting_ref_.get().model_setting_str);
  }

  return SUCCESS;
}

RESPONSE dstree::Filter::train(bool is_trial) {
  if (is_trained_ || config_.get().to_load_filters_) {
    return FAILURE;
  }

//  if (global_data_size_ < config_.get().filter_train_nexample_) {
//    spdlog::error("{:d} train examples collected; expected {:d}",
//                  global_data_size_, config_.get().filter_train_nexample_);
//
//    // TODO bind a wrapper for spdlog::shutdown and exit
//    spdlog::shutdown();
//    exit(FAILURE);
//  }

  if (!is_active_ && !is_trial) {
    spdlog::error("filter {:d} neither is_active nor is_trial; exit", id_);
    spdlog::shutdown();
    exit(FAILURE);
  }

  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

  if (config_.get().filter_remove_square_ && !is_distances_preprocessed_) {
    for (ID_TYPE i = 0; i < global_data_size_; ++i) {
      global_lnn_distances_[i] = sqrt(global_lnn_distances_[i]);
      global_bsf_distances_[i] = sqrt(global_bsf_distances_[i]);
    }

    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < global_data_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }

    if (local_data_size_ > 0) {
      for (ID_TYPE i = 0; i < local_data_size_; ++i) {
        local_lnn_distances_[i] = sqrt(local_lnn_distances_[i]);
      }
    }

    is_distances_preprocessed_ = true;
  }

#ifdef DEBUG
//#ifndef DEBUGGED
  if (!is_distances_logged) {
    if (!lb_distances_.empty()) {
      spdlog::debug("filter {:d} stream {:d} d_node{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(lb_distances_.data(), global_data_size_));
    }

    spdlog::debug("filter {:d} stream {:d} d_bsf{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_bsf_distances_.data(), global_data_size_));

    spdlog::debug("filter {:d} stream {:d} global d_nn{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_lnn_distances_.data(), global_data_size_));

    if (local_data_size_ > 0) {
      spdlog::debug("filter {:d} stream {:d} local d_nn{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(local_lnn_distances_.data(), local_data_size_));
    }

    is_distances_logged = true;
  }
//#endif
#endif

  ID_TYPE num_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
  ID_TYPE num_valid_examples = global_data_size_ - num_train_examples;

  torch::Tensor train_data, valid_data;
  torch::Tensor train_targets, valid_targets;

  if (local_data_size_ > 0) {
    assert(global_data_size_ == config_.get().filter_train_num_global_example_);
    assert(local_data_size_ == config_.get().filter_train_num_local_example_);

    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    ID_TYPE num_local_train_examples = local_data_size_ * config_.get().filter_train_val_split_;

    torch::Tensor global_train_data = global_queries_.get().index(
        {torch::indexing::Slice(0, num_train_examples)}).clone();
    torch::Tensor global_train_targets = torch::from_blob(global_lnn_distances_.data(),
                                                          num_global_train_examples,
                                                          torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    torch::Tensor local_train_data = torch::from_blob(local_queries_.data(),
                                                      {num_local_train_examples, config_.get().series_length_},
                                                      torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_train_targets = torch::from_blob(local_lnn_distances_.data(),
                                                         num_local_train_examples,
                                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    train_data = torch::cat({global_train_data, local_train_data}, 0);
    train_targets = torch::cat({global_train_targets, local_train_targets}, 0);

    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    ID_TYPE num_local_valid_examples = local_data_size_ - num_local_train_examples;

    torch::Tensor global_valid_data = global_queries_.get().index(
        {torch::indexing::Slice(num_train_examples, global_data_size_)}).clone();
    torch::Tensor global_valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples,
                                                          num_global_valid_examples,
                                                          torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    torch::Tensor local_valid_data = torch::from_blob(
        local_queries_.data() + num_local_train_examples * config_.get().series_length_,
        {num_local_valid_examples, config_.get().series_length_},
        torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_valid_targets = torch::from_blob(local_lnn_distances_.data() + num_local_train_examples,
                                                         num_local_valid_examples,
                                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = torch::cat({global_valid_data, local_valid_data}, 0);
    valid_targets = torch::cat({global_valid_targets, local_valid_targets}, 0);

    num_train_examples = num_global_train_examples + num_local_train_examples;
    num_valid_examples = num_global_valid_examples + num_local_valid_examples;

    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(valid_data.size(0) == num_train_examples && valid_targets.size(0) == num_train_examples);
  } else {
    assert(global_data_size_ == config_.get().filter_train_nexample_);

    train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    train_targets = torch::from_blob(global_lnn_distances_.data(),
                                     num_train_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = global_queries_.get().index({torch::indexing::Slice(
        num_train_examples, global_data_size_)}).clone();
    valid_targets = torch::from_blob(global_lnn_distances_.data() + num_train_examples,
                                     num_valid_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  }

  auto train_dataset = upcite::SeriesDataset(train_data, train_targets);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

//  auto valid_dataset = upcite::SeriesDataset(valid_data, valid_targets);
//  auto valid_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
//      valid_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

  // reuse validation examples as conformal examples
  ID_TYPE num_conformal_examples = num_valid_examples;
  torch::Tensor conformal_data = valid_data;
  torch::Tensor conformal_targets = valid_targets;

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("train thread {:d} node {:d} model {:d} n_train {:d} n_valid {:d} n_conformal {:d} n_batch {:d}",
                      trial_cache.thread_id_,
                      trial_sample_i,
                      model_i,
                      trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i]
        );
#endif
#endif

  // TODO instantiate the model according to the assigned model_setting_
  model_ = std::make_shared<dstree::MLPFilter>(config_.get().series_length_,
                                               config_.get().filter_dim_latent_,
                                               config_.get().filter_train_dropout_p_,
                                               config_.get().filter_leaky_relu_negative_slope_);

  model_->to(*device_);

  // for early termination
  std::unordered_map<std::string, torch::Tensor> best_model_state;
  VALUE_TYPE best_validation_loss = constant::MAX_VALUE;
  ID_TYPE best_validation_epoch = -1;

  torch::optim::SGD optimizer(model_->parameters(), config_.get().filter_train_learning_rate_);

  size_t initial_cooldown_epochs = config_.get().filter_train_nepoch_ / 2;
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(optimizer, initial_cooldown_epochs);

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

    { // evaluate
      VALUE_TYPE valid_loss = 0;

      c10::InferenceMode guard;
      model_->eval();

      torch::Tensor prediction = model_->forward(valid_data);

//      valid_loss = huber_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();
      valid_loss = mse_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif

      if (epoch > initial_cooldown_epochs) {
        if (best_validation_loss > valid_loss) {
          best_validation_loss = valid_loss;
          best_validation_epoch = epoch;

          for (const auto &pair : model_->named_parameters()) {
            best_model_state[pair.key()] = pair.value().clone();
          }
        }
      }

      upcite::optim::LR_RETURN_CODE return_code = lr_scheduler.check_step(valid_loss);
      if (return_code == upcite::optim::EARLY_STOP) {
        epoch = config_.get().filter_train_nepoch_;
      }
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} stream {:d} model {:s} t_losses = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} stream {:d} model {:s} v_losses = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));

  spdlog::debug("filter {:d} stream {:d} model {:s} restore from epoch {:d} of v_loss {:.4f}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                best_validation_epoch, best_validation_loss);
#endif

  c10::InferenceMode guard;

  for (auto &pair : best_model_state) {
    model_->named_parameters()[pair.first].detach_();
    model_->named_parameters()[pair.first].copy_(pair.second);
  }

  model_->eval();

  auto prediction = model_->forward(global_queries_).detach().cpu();
  auto *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();

  global_pred_distances_.insert(global_pred_distances_.end(), predictions_array, predictions_array + global_data_size_);

#ifdef DEBUG
  spdlog::info("filter {:d} stream {:d} model {:s} global d_pred{:s} = {:s}",
               id_, stream_id, model_setting_ref_.get().model_setting_str,
               config_.get().filter_remove_square_ ? "" : "_sq",
               upcite::array2str(predictions_array, global_data_size_));
#endif

  if (config_.get().filter_is_conformal_) {
    fit_conformal_predictor(is_trial);
  }

//  net->to(torch::Device(torch::kCPU));
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!is_trial) {
    is_trained_ = true;
  }

  return SUCCESS;
}

RESPONSE dstree::Filter::collect_running_info(MODEL_SETTING &model_setting) {
  model_setting_ref_ = model_setting;

  // TODO instantiate the model according to the assigned model_setting_
  model_ = std::make_shared<dstree::MLPFilter>(config_.get().series_length_,
                                               config_.get().filter_dim_latent_,
                                               config_.get().filter_train_dropout_p_,
                                               config_.get().filter_leaky_relu_negative_slope_);

  model_->to(*device_);

  c10::InferenceMode guard;
  model_->eval();

  if (config_.get().filter_is_conformal_ && !conformal_predictor_->is_fitted()) {
    fit_conformal_predictor(false, true);
  }

  size_t memory_size = 0;

  for (const auto &parameter : model_->parameters()) {
    memory_size += parameter.nbytes() * 2; // x2 inflates to cater the peak
  }

  for (const auto &buffer : model_->buffers()) {
    memory_size += buffer.nbytes();
  }

  model_setting_ref_.get().gpu_mem_mb = static_cast<VALUE_TYPE>(memory_size) / (1024 * 1024);

  auto trial_query = global_queries_.get().index({torch::indexing::Slice(0, 1)}).clone();
  auto trial_predictions = make_reserved<VALUE_TYPE>(config_.get().filter_trial_iterations_);

  auto start = std::chrono::high_resolution_clock::now();

  for (ID_TYPE trial_i = 0; trial_i < config_.get().filter_trial_iterations_; ++trial_i) {
    auto pred = model_->forward(trial_query).item<VALUE_TYPE>();

    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
    }

    if (config_.get().filter_remove_square_) {
      trial_predictions.push_back(pred * pred);
    } else {
      trial_predictions.push_back(pred);
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::info("filter trial model {:s} {:d} iters = {:d}mus",
               model_setting_ref_.get().model_setting_str,
               config_.get().filter_trial_iterations_,
               duration.count());
#endif
#endif

  model_setting_ref_.get().gpu_ms_per_query =
      static_cast<double_t>(duration.count()) / static_cast<double_t>(config_.get().filter_trial_iterations_);

#ifdef DEBUG
  spdlog::info("filter trial model {:s} gpu mem = {:.3f}MB, time = {:.6f}mus",
               model_setting_ref_.get().model_setting_str,
               model_setting_ref_.get().gpu_mem_mb,
               model_setting_ref_.get().gpu_ms_per_query);
#endif

  return SUCCESS;
}

VALUE_TYPE dstree::Filter::infer(torch::Tensor &query_series) const {
#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("filter {:d} {:b} device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                device_->str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  spdlog::debug("filter {:d} {:b} query device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                query_series.device().str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);

  auto paras = model_->parameters();
  for (ID_TYPE i = 0; i < paras.size(); ++i) {
    spdlog::debug("filter {:d} {:b} model_p_{:d} device {:s}, requested {:b}:{:d}",
                  id_, is_trained_,
                  i, paras[i].device().str(),
                  config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  }
#endif
#endif

  if (is_trained_) {
    c10::InferenceMode guard;

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
  node_fos.write(reinterpret_cast<const char *>(&global_data_size_), sizeof(ID_TYPE));

  // TODO condense size indicators into a bitmap, as they all = train_size_
  // bsf_distances_
  ID_TYPE size_placeholder = global_bsf_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_bsf_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_bsf_distances_.data()),
                   sizeof(VALUE_TYPE) * global_bsf_distances_.size());
  }

  // nn_distances_
  size_placeholder = global_lnn_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_lnn_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_lnn_distances_.data()),
                   sizeof(VALUE_TYPE) * global_lnn_distances_.size());
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
  size_placeholder = global_pred_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_pred_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_pred_distances_.data()),
                   sizeof(VALUE_TYPE) * global_pred_distances_.size());
  }

  node_fos.write(reinterpret_cast<const char *>(&local_data_size_), sizeof(ID_TYPE));

  if (local_data_size_ > 0) {
    // local_queries_
    size_placeholder = local_queries_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
    if (!local_queries_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_queries_.data()),
                     sizeof(VALUE_TYPE) * local_queries_.size());
    }

    // local_lnn_distances_
    size_placeholder = local_lnn_distances_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
    if (!local_lnn_distances_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_lnn_distances_.data()),
                     sizeof(VALUE_TYPE) * local_lnn_distances_.size());
    }

  }

  if (is_active_) {
    size_placeholder = model_setting_ref_.get().model_setting_str.size();
  } else {
    size_placeholder = 0;
  }
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (is_active_) {
    node_fos.write(reinterpret_cast<const char *>(model_setting_ref_.get().model_setting_str.data()),
                   sizeof(model_setting_ref_.get().model_setting_str));
  }

  ID_TYPE is_trained_placeholder = 0;
  if (is_trained_) {
    is_trained_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_trained_placeholder), sizeof(ID_TYPE));
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
  global_data_size_ = ifs_id_buf[0];

  // bsf_distances_
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_bsf_distances_.insert(global_bsf_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // nn_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_lnn_distances_.insert(global_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
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
    global_pred_distances_.insert(global_pred_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // local_data_size_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  local_data_size_ = ifs_id_buf[0];

  // local_queries_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    local_queries_.insert(local_queries_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }

  // local_lnn_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    local_lnn_distances_.insert(local_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
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
      model_setting_ref_ = std::ref(model_setting_);

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

    if (config_.get().filter_infer_is_gpu_) {
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }

    // TODO instantiate the model according to the setting
    model_ = std::make_unique<dstree::MLPFilter>(config_.get().series_length_,
                                                 config_.get().filter_dim_latent_,
                                                 config_.get().filter_train_dropout_p_,
                                                 config_.get().filter_leaky_relu_negative_slope_);
    torch::load(model_, model_filepath);
    model_->to(*device_);

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
    conformal_predictor_->load(node_ifs, ifs_buf);

    if (config_.get().filter_is_conformal_) {
      // TODO check compatibility between the loaded setting and the new setting
      fit_conformal_predictor();
    }
  }

  return SUCCESS;
}

VALUE_TYPE dstree::Filter::get_node_summarization_pruning_frequency() const {
  if (lb_distances_.empty() || lb_distances_.size() != global_bsf_distances_.size()) {
    return 0;
  }

  ID_TYPE pruned_counter = 0;
  for (ID_TYPE i = 0; i < lb_distances_.size(); ++i) {
    if (lb_distances_[i] > global_bsf_distances_[i]) {
      pruned_counter += 1;
    }
  }

  return static_cast<VALUE_TYPE>(pruned_counter) / static_cast<VALUE_TYPE>(lb_distances_.size());
}

VALUE_TYPE upcite::dstree::Filter::get_val_pruning_ratio() const {
  ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;

  VALUE_TYPE abs_error_interval = get_abs_error_interval();
  ID_TYPE pruned_counter = 0;

  for (ID_TYPE example_i = num_global_train_examples; example_i < global_data_size_; ++example_i) {
    if (global_pred_distances_[example_i] - abs_error_interval > global_bsf_distances_[example_i]) {
      pruned_counter += 1;
    }
  }

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("filter {:d} model {:s} err = {:.3f}, {:d} / {:d} (+{:d}) = {:.3f}",
                id_, model_setting_ref_.get().model_setting_str,
                abs_error_interval,
                pruned_counter,
                num_conformal_examples,
                num_train_examples,
                static_cast<VALUE_TYPE>(pruned_counter) / num_conformal_examples);
#endif
#endif

  ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
  return static_cast<VALUE_TYPE>(pruned_counter) / num_global_valid_examples;
}
