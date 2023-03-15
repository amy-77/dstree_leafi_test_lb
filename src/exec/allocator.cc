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
    config_(config) {
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
      spdlog::error("allocator: default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator: > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      for (auto &filter_info : filter_infos_) {
        filter_info.model_setting = candidate_model_settings_[0];
      }
    }
  }

  // TODO uncomment after thalia gets restarted
//  size_t gpu_free_bytes_, gpu_total_bytes_;
//  cuMemGetInfo(&gpu_free_bytes_, &gpu_total_bytes_);
//  VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes_) / 1000 / 1000;
//  if (gpu_free_mb < available_gpu_memory_mb_) {
//    spdlog::error("GPU: required {:.3f}mb is not available; down to all free {:.3f}mb",
//                  available_gpu_memory_mb_, gpu_free_mb);
//    available_gpu_memory_mb_ = gpu_free_mb;
//  }

  VALUE_TYPE allocated_gpu_memory_mb = 0;
  for (auto &filter_info : filter_infos_) {
    if (allocated_gpu_memory_mb >= available_gpu_memory_mb_) {
      break;
    }

    if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
      allocated_gpu_memory_mb += model_gpu_memory_fingerprint_mb_[filter_info.model_setting.model_setting_str];
    }
  }
  spdlog::info("GPU: allocated {:.3f}mb of available {:.3f}mb",
               allocated_gpu_memory_mb, available_gpu_memory_mb_);

  return SUCCESS;
}
