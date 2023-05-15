//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "allocator.h"

#include <spdlog/spdlog.h>
#include <cuda.h>

namespace dstree = upcite::dstree;

dstree::Allocator::Allocator(dstree::Config &config,
                             ID_TYPE nfilters) :
    config_(config),
    is_recall_calculated_(false) {
  cpu_sps_ = -1;

  // TODO test overhead
  cpu_overhead_pn_ = 0;
  gpu_overhead_pn_ = -1;

  available_gpu_memory_mb_ = config.filter_max_gpu_memory_mb_;

  // TODO support model setting list
  candidate_model_settings_.emplace_back(config.filter_model_setting_str_);

  if (nfilters > 0) {
    filter_infos_.reserve(nfilters);
  }
}

RESPONSE dstree::Allocator::push_instance(const dstree::FilterInfo &filter_info) {
  filter_infos_.push_back(filter_info);

  return SUCCESS;
}

RESPONSE dstree::Allocator::evaluate() {
  for (auto &filter_info : filter_infos_) {
    filter_info.score = constant::MIN_VALUE;

    for (auto candidate_model_setting_ : candidate_model_settings_) {

      // TODO support model in cpu
      double_t amortized_gpu_sps =
          static_cast<double_t>(candidate_model_setting_.gpu_sps + gpu_overhead_pn_ - cpu_overhead_pn_)
              / static_cast<double_t>(filter_info.node_.get().get_size());

      if (amortized_gpu_sps > cpu_sps_) {
        spdlog::error("model {:s} slower than cpu: {:f} > {:f}",
                      candidate_model_setting_.model_setting_str, amortized_gpu_sps, cpu_sps_);
      }

      auto gain = static_cast<VALUE_TYPE>(static_cast<double_t>(filter_info.node_.get().get_size())
          * static_cast<double_t>((1 - filter_info.external_pruning_probability_) * filter_info.pruning_probability_)
          * static_cast<double_t>(cpu_sps_ - amortized_gpu_sps));

      if (gain > filter_info.score) {
        filter_info.score = gain;
        filter_info.model_setting = candidate_model_setting_;
      }
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::assign() {
  if (config_.get().filter_allocate_is_gain_) {
    evaluate();

    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterScore);
  } else { // default
    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

    if (config_.get().filter_model_setting_str_.empty() || candidate_model_settings_.empty()) {
      spdlog::error("allocator default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      for (auto &filter_info : filter_infos_) {
        filter_info.model_setting = candidate_model_settings_[0];
      }
    }
  }

  size_t gpu_free_bytes_, gpu_total_bytes_;
  cuMemGetInfo(&gpu_free_bytes_, &gpu_total_bytes_);
  VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes_) / 1000 / 1000;

  if (gpu_free_mb < available_gpu_memory_mb_) {
    spdlog::error("allocator required {:.3f}mb is not available; down to all free {:.3f}mb",
                  available_gpu_memory_mb_, gpu_free_mb);
    available_gpu_memory_mb_ = gpu_free_mb;
  } else {
    spdlog::info("allocator requested {:.3f}mb; {:.3f}mb available",
                 available_gpu_memory_mb_, gpu_free_mb);
  }

  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;

  for (auto &filter_info : filter_infos_) {
    if (allocated_gpu_memory_mb >= available_gpu_memory_mb_) {
      break;
    }

    if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
      allocated_gpu_memory_mb += model_gpu_memory_fingerprint_mb_[filter_info.model_setting.model_setting_str];
      allocated_filters_count += 1;
    }
  }

  spdlog::info("allocator assigned {:d} models of {:.3f}mb/{:.3f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);

  return SUCCESS;
}

RESPONSE dstree::Allocator::set_confidence_from_recall() {
  if (!is_recall_calculated_) {
    auto num_train_examples =
        static_cast<ID_TYPE>(config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_);
    ID_TYPE num_valid_examples = config_.get().filter_train_nexample_ - num_train_examples;
    ID_TYPE num_conformal_examples = num_valid_examples * config_.get().filter_conformal_train_val_split_;

    auto nn_distances = upcite::make_reserved<VALUE_TYPE>(num_conformal_examples);
    auto nn_filter_ids = upcite::make_reserved<ID_TYPE>(num_conformal_examples);

    for (ID_TYPE i = 0; i < num_conformal_examples; ++i) {
      nn_distances[i] = constant::MAX_VALUE;
    }

    for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
      for (ID_TYPE filter_i = 0; filter_i < filter_infos_.size(); ++filter_i) {
//    if (filter_infos_[i].node_.get().has_active_filter()) { // nn should be searched from all nodes
        VALUE_TYPE local_nn = filter_infos_[filter_i].node_.get().get_filter_nn_distance(num_train_examples + query_i);

        if (local_nn < nn_distances[query_i]) {
          nn_distances[query_i] = local_nn;
          nn_filter_ids[query_i] = filter_i;
        }
      }
    }

    // two sentry points: recall at small error (recall_at_0_error, 0_error), recall at large error (0.999999, 42)
    validation_recalls_.reserve(num_conformal_examples + 2);

    // recall at validation error intervals
    for (ID_TYPE sorted_error_i = 0; sorted_error_i < num_conformal_examples + 2; ++sorted_error_i) {
      ID_TYPE hit_count = 0;

      for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        std::reference_wrapper<dstree::Node> target_node = filter_infos_[nn_filter_ids[query_i]].node_;

        VALUE_TYPE abs_error_interval = target_node.get().get_filter_abs_error_interval_by_pos(sorted_error_i);

        VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(num_train_examples + query_i);
        VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(num_train_examples + query_i);

        if (pred_distance - abs_error_interval <= bsf_distance) {
          hit_count += 1;
        }
      }

      validation_recalls_.push_back(static_cast<ERROR_TYPE>(hit_count) / num_conformal_examples);
    }

    // to adjust the recalls to be strictly increasing
    validation_recalls_[validation_recalls_.size() - 1] = 1 - constant::EPSILON_GAP;
    for (ID_TYPE backtrace_i = validation_recalls_.size() - 2; backtrace_i >= 0; --backtrace_i) {
      if (validation_recalls_[backtrace_i] > validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP) {
        validation_recalls_[backtrace_i] = validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP;
      }
    }

    spdlog::debug("allocator recalls = {:s}",
                  upcite::array2str(validation_recalls_.data(), validation_recalls_.size()));

    is_recall_calculated_ = true;

    if (config_.get().filter_conformal_is_smoothen_) {
      for (auto &filter_info : filter_infos_) {
        if (filter_info.node_.get().has_active_filter()) {
          filter_info.node_.get().fit_filter_conformal_spline(validation_recalls_);
        }
      }
    }
  }

  if (config_.get().filter_conformal_is_smoothen_) {
    for (auto &filter_info : filter_infos_) {
      if (filter_info.node_.get().has_active_filter()) {
        filter_info.node_.get().set_filter_abs_error_interval_by_recall(config_.get().filter_conformal_recall_);

        spdlog::info("allocator node {:d} abs_error {:.3f} at {:.4f}",
                     filter_info.node_.get().get_id(),
                     filter_info.node_.get().get_filter_abs_error_interval(),
                     config_.get().filter_conformal_recall_);
      }
    }
  } else {
    ID_TYPE last_recall_i = validation_recalls_.size() - 1;

    for (ID_TYPE recall_i = validation_recalls_.size() - 2; recall_i >= 0; --recall_i) {
      if (validation_recalls_[recall_i] < config_.get().filter_conformal_recall_) {
        last_recall_i = recall_i + 1;

        spdlog::info("allocator reached recall {:.3f} with error_i {:.3f} ({:d}/{:d}, 2 sentries included)",
                     validation_recalls_[last_recall_i],
                     static_cast<VALUE_TYPE>(last_recall_i) / validation_recalls_.size(),
                     last_recall_i,
                     validation_recalls_.size());
        break;
      }
    }

    for (auto &filter_info : filter_infos_) {
      if (filter_info.node_.get().has_active_filter()) {
        filter_info.node_.get().set_filter_abs_error_interval_by_pos(last_recall_i);
      }
    }
  }

  return SUCCESS;
}
