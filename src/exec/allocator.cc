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

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("allocator nn_distances = {:s}",
                    upcite::array2str(nn_distances.data(), num_conformal_examples));
  spdlog::debug("allocator nn_filter_ids = {:s}",
                upcite::array2str(nn_filter_ids.data(), num_conformal_examples));

  for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
    spdlog::debug("allocator query_i {:d} nn {:.5f} filter_i {:d} filter_nn {:.5f} filter_bsf {:.5f} filter_pred {:.5f}",
                  query_i,
                  nn_distances[query_i],
                  nn_filter_ids[query_i],
                  filter_infos_[nn_filter_ids[query_i]].node_.get().get_filter_nn_distance(num_train_examples + query_i),
                  filter_infos_[nn_filter_ids[query_i]].node_.get().get_filter_bsf_distance(num_train_examples + query_i),
                  filter_infos_[nn_filter_ids[query_i]].node_.get().get_filter_pred_distance(num_train_examples + query_i)
                  );
  }
#endif
#endif

  VALUE_TYPE target_recall_level = config_.get().filter_conformal_recall_;
  VALUE_TYPE current_recall = 1, last_recall = 1;
  ID_TYPE last_pos = num_conformal_examples;

  for (ID_TYPE confidence_pos = num_conformal_examples - 1; confidence_pos >= 0; --confidence_pos) {
    ID_TYPE miss_count = 0;

    for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
      std::reference_wrapper<dstree::Node> target_node = filter_infos_[nn_filter_ids[query_i]].node_;

      VALUE_TYPE confidence_half_interval =
          target_node.get().get_filter_confidence_half_interval_by_pos(confidence_pos);

      VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(num_train_examples + query_i);
      VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(num_train_examples + query_i);

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator confdn {:d} query {:d} node {:d} nn {:.7f} bsf {:.7f} pred {:.7f} intvl {:.7f}",
                      confidence_pos,
                      query_i,
                      target_node.get().get_id(),
                      target_node.get().get_filter_nn_distance(num_train_examples + query_i),
                      bsf_distance,
                      pred_distance,
                      confidence_half_interval);
#endif
#endif

      if (pred_distance - confidence_half_interval > bsf_distance) {
        miss_count += 1;
      }
    }

    current_recall = static_cast<VALUE_TYPE>(num_conformal_examples - miss_count) / num_conformal_examples;

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("allocator required recall {:.2f} reached {:.3f} with confidence {:.3f} ({:d}/{:d})",
               target_recall_level, current_recall,
               static_cast<VALUE_TYPE>(confidence_pos + 1) / num_conformal_examples,
               confidence_pos + 1, num_conformal_examples);
#endif
#endif

    if (current_recall < target_recall_level) {
      break;
    } else {
      last_recall = current_recall;
      last_pos = confidence_pos;
    }
  }

  // training pruning ratio could be calculated from the log
  spdlog::info("allocator reached recall {:.3f} with confidence {:.3f} ({:d}/{:d})",
               last_recall,
               static_cast<VALUE_TYPE>(last_pos + 1) / num_conformal_examples,
               last_pos + 1,
               num_conformal_examples);

  for (auto &filter_info : filter_infos_) {
    if (filter_info.node_.get().has_active_filter()) {
      filter_info.node_.get().set_filter_confidence_half_interval_by_pos(last_pos);
    }
  }

  return SUCCESS;
}
