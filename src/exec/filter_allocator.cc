//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "filter_allocator.h"

#include <random>
#include <immintrin.h>

#include <spdlog/spdlog.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "vec.h"
#include "distance.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Allocator::Allocator(dstree::Config &config,
                             ID_TYPE nfilters) :
    config_(config),
    is_recall_calculated_(false),
    node_size_threshold_(0) {

  if (config_.get().filter_infer_is_gpu_) {
    if (torch::cuda::is_available()) {
      cudaSetDevice(config_.get().filter_device_id_);

      size_t gpu_free_bytes_, gpu_total_bytes_;
      cuMemGetInfo(&gpu_free_bytes_, &gpu_total_bytes_);
      VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes_) / 1024 / 1024;

      if (gpu_free_mb < config.filter_max_gpu_memory_mb_) {
        if (gpu_free_mb > 1) {
          spdlog::error("allocator required {:.3f}mb is not available; down to all free {:.3f}mb",
                        config.filter_max_gpu_memory_mb_, gpu_free_mb);

          available_gpu_memory_mb_ = gpu_free_mb;
        } else {
          spdlog::error("allocator only {:.3f}mb gpu memory is free; exit", gpu_free_mb);
          spdlog::shutdown();
          exit(FAILURE);
        }
      } else {
        spdlog::info("allocator requested {:.3f}mb; {:.3f}mb available",
                     config.filter_max_gpu_memory_mb_, gpu_free_mb);

        available_gpu_memory_mb_ = config.filter_max_gpu_memory_mb_;
      }
    } else {
      spdlog::error("allocator gpu unavailable");
      spdlog::shutdown();
      exit(FAILURE);
    }
  }

  // TODO support model setting list
  candidate_model_settings_.emplace_back(config.filter_model_setting_str_);

  if (nfilters > 0) {
    filter_infos_.reserve(nfilters);
  }

  measure_cpu();
  assert(config_.get().filter_infer_is_gpu_);
  measure_gpu();

  node_size_threshold_ = constant::MAX_ID;
  for (ID_TYPE candidate_model_i = 0; candidate_model_i < candidate_model_settings_.size(); ++candidate_model_i) {
    ID_TYPE current_node_size_threshold =
        candidate_model_settings_[candidate_model_i].gpu_ms_per_query / cpu_ms_per_series_;
    if (current_node_size_threshold < node_size_threshold_) {
      node_size_threshold_ = current_node_size_threshold;
    }
  }
#ifdef DEBUG
  spdlog::info("allocator node size threshold = {:d}; default {:d}",
               node_size_threshold_, config_.get().filter_default_node_size_threshold_);
#endif

  if (node_size_threshold_ < config_.get().filter_default_node_size_threshold_) {
    node_size_threshold_ = config_.get().filter_default_node_size_threshold_;

#ifdef DEBUG
    spdlog::info("allocator node size threshold rectified to default {:d}", node_size_threshold_);
#endif
  }
}

RESPONSE dstree::Allocator::push_filter_info(const FilterInfo &filter_info) {
  filter_infos_.push_back(filter_info);

  return SUCCESS;
}

struct TrialCache {
  TrialCache(dstree::Config &config,
             ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::vector<upcite::MODEL_SETTING> &candidate_model_settings,
             std::vector<dstree::FilterInfo> &filter_infos,
             ID_TYPE trial_nnode,
             ID_TYPE trial_nmodel,
             std::vector<ID_TYPE> &sampled_filter_idx,
             std::vector<VALUE_TYPE> &filter_pruning_ratios,
             ID_TYPE *trial_sample_i_ptr,
             pthread_mutex_t *sample_idx_mutex) :
      config_(config),
      thread_id_(thread_id),
      stream_(stream),
      candidate_model_settings_ref_(candidate_model_settings),
      filter_infos_ref_(filter_infos),
      trial_nnode_(trial_nnode),
      trial_nmodel_(trial_nmodel),
      sampled_filter_idx_ref_(sampled_filter_idx),
      filter_pruning_ratios_ref_(filter_pruning_ratios),
      trial_sample_i_ptr_(trial_sample_i_ptr),
      sample_idx_mutex_(sample_idx_mutex) {}
  ~TrialCache() = default;

  std::reference_wrapper<dstree::Config> config_;

  ID_TYPE thread_id_;

  at::cuda::CUDAStream stream_;

  std::reference_wrapper<std::vector<upcite::MODEL_SETTING>> candidate_model_settings_ref_;
  std::reference_wrapper<std::vector<dstree::FilterInfo>> filter_infos_ref_;

  ID_TYPE trial_nnode_;
  ID_TYPE trial_nmodel_;

  std::reference_wrapper<std::vector<ID_TYPE>> sampled_filter_idx_ref_;
  std::reference_wrapper<std::vector<VALUE_TYPE>> filter_pruning_ratios_ref_;

  ID_TYPE *trial_sample_i_ptr_;
  pthread_mutex_t *sample_idx_mutex_;
};

void trial_thread_F(TrialCache &trial_cache) {
  at::cuda::setCurrentCUDAStream(trial_cache.stream_);
  at::cuda::CUDAStreamGuard guard(trial_cache.stream_); // compiles with libtorch-gpu

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("allocator thread {:d}", trial_cache.thread_id_);
  spdlog::debug("allocator candidate_model_settings_ref_.get().size() {:d}",
                trial_cache.candidate_model_settings_ref_.get().size());
  spdlog::debug("allocator filter_infos_ref_.get().size() {:d}", trial_cache.filter_infos_ref_.get().size());
  spdlog::debug("allocator trial_nnode_ {:d}", trial_cache.trial_nnode_);
  spdlog::debug("allocator trial_nmodel_ {:d}", trial_cache.trial_nmodel_);
  spdlog::debug("allocator sampled_filter_idx_ref_.get().size() {:d}",
                trial_cache.sampled_filter_idx_ref_.get().size());
  spdlog::debug("allocator filter_pruning_ratios_ref_.get().size() {:d}",
                trial_cache.filter_pruning_ratios_ref_.get().size());
  spdlog::debug("allocator trial_sample_i_ {:d}", *trial_cache.trial_sample_i_ptr_);
#endif
#endif

  while (true) {
    pthread_mutex_lock(trial_cache.sample_idx_mutex_);

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("allocator thread {:d} locked, *trial_cache.trial_sample_i_ptr_ = {:d}",
                  trial_cache.thread_id_,
                  *trial_cache.trial_sample_i_ptr_);
#endif
#endif

    if ((*trial_cache.trial_sample_i_ptr_) >= trial_cache.trial_nnode_) {
      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      break;
    } else {
      // iterate over nodes (check all models for this node)
      // TODO iterate over sampled [node, model] pairs
      ID_TYPE trial_sample_i = *trial_cache.trial_sample_i_ptr_;
      *trial_cache.trial_sample_i_ptr_ = trial_sample_i + 1;

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} to unlock; trial_sample_i = {:d}, *trial_cache.trial_sample_i_ptr_ = {:d}",
                    trial_cache.thread_id_,
                    trial_sample_i,
                    *trial_cache.trial_sample_i_ptr_);
#endif
#endif

      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      ID_TYPE filter_sample_pos = trial_cache.sampled_filter_idx_ref_.get()[trial_sample_i];

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} sampled_filter_id = {:d}",
                    trial_cache.thread_id_, filter_sample_pos);
#endif
#endif

      std::reference_wrapper<dstree::FilterInfo> filter_info = trial_cache.filter_infos_ref_.get()[filter_sample_pos];
      auto filter_ref = filter_info.get().node_.get().get_filter();

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} check node {:d}",
                    trial_cache.thread_id_,
                    filter_ref.get().get_id()
      );
#endif
#endif

      for (ID_TYPE model_i = 0; model_i < trial_cache.trial_nmodel_; ++model_i) {
        auto &candidate_model_setting = trial_cache.candidate_model_settings_ref_.get()[model_i];

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} check model {:s} on node {:d}",
                      trial_cache.thread_id_,
                      candidate_model_setting.model_setting_str,
                      filter_ref.get().get_id()
        );
#endif
#endif

        filter_ref.get().trigger_trial(candidate_model_setting);
        filter_ref.get().train(true);

        // 2-d array of [no. models, no. nodes]
        trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i] =
            filter_ref.get().get_val_pruning_ratio();

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} node {:d} model {:d} pruning ratio = {:.3f}",
                      trial_cache.thread_id_,
                      trial_sample_i,
                      model_i,
                      trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i]
        );
#endif
#endif
      }
    }
  }
}

RESPONSE dstree::Allocator::trial_collect_mthread() {
  std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

  ID_TYPE end_i_exclusive = filter_infos_.size();
  while (end_i_exclusive > 1 && filter_infos_[end_i_exclusive - 1].node_.get().get_size()
      < config_.get().filter_default_node_size_threshold_) {
    end_i_exclusive -= 1;
  }

  ID_TYPE offset = 0;
  ID_TYPE step = end_i_exclusive / config_.get().filter_trial_nnode_;

  auto sampled_filter_idx = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE sample_i = 0; sample_i < config_.get().filter_trial_nnode_; ++sample_i) {
    sampled_filter_idx.push_back(offset + sample_i * step);
  }

  // 2-d array of [no. models, no. nodes]
  auto filter_pruning_ratios = upcite::make_reserved<VALUE_TYPE>(
      config_.get().filter_trial_nnode_ * candidate_model_settings_.size());

  for (ID_TYPE i = 0; i < config_.get().filter_trial_nnode_ * candidate_model_settings_.size(); ++i) {
    filter_pruning_ratios.push_back(0);
  }

  std::vector<std::unique_ptr<TrialCache>> trial_caches;
  std::unique_ptr<pthread_mutex_t> sample_idx_mutex = std::make_unique<pthread_mutex_t>();
  ID_TYPE trial_sample_i = 0;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    spdlog::info("trial thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
                 thread_id,
                 static_cast<ID_TYPE>(new_stream.id()),
                 static_cast<ID_TYPE>(new_stream.query()),
                 static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu

    trial_caches.emplace_back(std::make_unique<TrialCache>(config_,
                                                           thread_id,
                                                           std::move(new_stream),
                                                           std::ref(candidate_model_settings_),
                                                           std::ref(filter_infos_),
                                                           config_.get().filter_trial_nnode_,
                                                           candidate_model_settings_.size(),
                                                           std::ref(sampled_filter_idx),
                                                           std::ref(filter_pruning_ratios),
                                                           &trial_sample_i,
                                                           sample_idx_mutex.get()));
  }

  std::vector<std::thread> threads;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads.emplace_back(trial_thread_F, std::ref(*trial_caches[thread_id]));
  }

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads[thread_id].join();
  }

#ifdef DEBUG
  auto sampled_filter_ids = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE filter_i = 0; filter_i < config_.get().filter_trial_nnode_; ++filter_i) {
    sampled_filter_ids.push_back(filter_infos_[sampled_filter_idx[filter_i]].node_.get().get_id());
  }

  spdlog::info("allocator sampled node ids = {:s}",
               upcite::array2str(sampled_filter_ids.data(), sampled_filter_ids.size()));
#endif

#ifdef DEBUG
  spdlog::info("allocator trial pruning ratios = {:s}",
               upcite::array2str(filter_pruning_ratios.data(), filter_pruning_ratios.size()));
#endif

  for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
    VALUE_TYPE mean = 0;

    for (ID_TYPE sample_i = 0; sample_i < sampled_filter_idx.size(); ++sample_i) {
      // 2-d array of [no. models, no. nodes]
      mean += filter_pruning_ratios[sampled_filter_idx.size() * model_i + sample_i];
    }

    candidate_model_settings_[model_i].pruning_prob = mean / sampled_filter_idx.size();

#ifdef DEBUG
    spdlog::info("allocator model {:s} pruning ratio = {:.3f}",
                 candidate_model_settings_[model_i].model_setting_str,
                 candidate_model_settings_[model_i].pruning_prob);
#endif
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_cpu() {
  // test cpu_ms_per_series_
  auto batch_nbytes = static_cast<ID_TYPE>(
      sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().leaf_max_nseries_;
  auto trial_batch = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

  auto distances = make_reserved<VALUE_TYPE>(config_.get().leaf_max_nseries_);

  if (config_.get().on_disk_) {
    // credit to https://stackoverflow.com/a/19728404
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<ID_TYPE> uni_i_d(0, config_.get().db_nseries_ - config_.get().leaf_max_nseries_);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      std::ifstream db_fin;
      db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);

      ID_TYPE batch_bytes_offset = static_cast<ID_TYPE>(
          sizeof(VALUE_TYPE)) * config_.get().series_length_ * uni_i_d(rng);

      db_fin.seekg(batch_bytes_offset);
      db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }

      db_fin.close();
      distances.clear();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  } else {
    std::ifstream db_fin;
    db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      distances.clear();

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    db_fin.close();

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  }

#ifdef DEBUG
  spdlog::info("allocator trial cpu time = {:.6f}mus", cpu_ms_per_series_);
#endif

  free(trial_batch);
  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_gpu() {
  if (torch::cuda::is_available()) {
    bool measure_required = false;
    for (auto const &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        measure_required = true;
      }
    }

    if (!measure_required) {
      return SUCCESS;
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);
    ID_TYPE query_nbytes = sizeof(VALUE_TYPE) * config_.get().series_length_;
    auto random_input = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

    for (ID_TYPE i = 0; i < config_.get().series_length_; ++i) {
      random_input[i] = dist(e2);
    }

    auto input_tsr_ = torch::from_blob(random_input,
                                       {1, config_.get().series_length_},
                                       torch::TensorOptions().dtype(TORCH_VALUE_TYPE));

    std::unique_ptr<torch::Device> device = nullptr;
    if (config_.get().filter_infer_is_gpu_) {
      device = std::make_unique<torch::Device>(torch::kCUDA,
                                               static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device = std::make_unique<torch::Device>(torch::kCPU);
    }
    input_tsr_ = input_tsr_.to(*device);

    auto trial_filter = std::make_unique<dstree::Filter>(config_, -1, input_tsr_);

    for (auto &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        trial_filter->collect_running_info(model_setting_ref);
      }
    }

    free(random_input);
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::evaluate() {
  // test candidate_model_setting_.pruning_prob
  trial_collect_mthread();

  // calculate gain for [node_i, model_i]
  filter_ids_.reserve(filter_infos_.size());
  gains_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());
  mem_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());

  for (ID_TYPE filter_i = 0; filter_i < filter_infos_.size(); ++filter_i) {
    auto &filter_info = filter_infos_[filter_i];
    filter_info.score = 0;

    filter_ids_.push_back(filter_info.node_.get().get_id());

    for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
      auto &candidate_model_setting_ = candidate_model_settings_[model_i];

      // TODO support model in cpu
      double_t amortized_gpu_sps = static_cast<double_t>(candidate_model_setting_.gpu_ms_per_query)
          / static_cast<double_t>(filter_info.node_.get().get_size());

      if (amortized_gpu_sps > cpu_ms_per_series_) {
        spdlog::error("allocator model {:s} slower than cpu: {:f} > {:f}",
                      candidate_model_setting_.model_setting_str, amortized_gpu_sps, cpu_ms_per_series_);
      }

      auto gain = static_cast<VALUE_TYPE>(static_cast<double_t>(filter_info.node_.get().get_size())
          * static_cast<double_t>((1 - filter_info.external_pruning_probability_)
              * candidate_model_setting_.pruning_prob)
          * (cpu_ms_per_series_ - amortized_gpu_sps));

      if (gain < 0) {
        // forbid harmful plans
        gains_matrix_.push_back(0);
        mem_matrix_.push_back(available_gpu_memory_mb_ + 1);
      } else {
        gains_matrix_.push_back(gain);
        mem_matrix_.push_back(candidate_model_setting_.gpu_mem_mb);
      }

      if (gain > filter_info.score) {
        filter_info.score = gain;
        filter_info.model_setting = candidate_model_setting_;
      }
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::assign() {
  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;

  if (config_.get().filter_allocate_is_gain_) {
    evaluate();

    if (candidate_model_settings_.size() == 1) {
      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterScore);

      if (filter_infos_[0].score <= 0) {
        spdlog::error("allocator gain-based allocation failed; revert to size-based allocation");

        for (auto &filter_info : filter_infos_) {
          if (filter_info.node_.get().get_size() > config_.get().filter_node_size_threshold_
              && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
            if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
              allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
              allocated_filters_count += 1;
            }
          } else {
            break;
          }
        }
      } else {
        for (auto &filter_info : filter_infos_) {
          if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_
              || filter_info.score <= 0) {
            break;
          }

          if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        }
      }
    } else {
      // TODO knapsack solver
    }
  } else { // default: implant the default model to all leaf nodes
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
        if (filter_info.node_.get().get_size() > config_.get().filter_node_size_threshold_
            && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        } else {
          break;
        }
      }
    }
  }

  spdlog::info("allocator assigned {:d} models of {:.3f}mb/{:.3f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}

RESPONSE dstree::Allocator::reassign() {
  if (candidate_model_settings_.size() != 1) {
    spdlog::error("allocator reallocation only supports single candidate");
    return FAILURE;
  }

  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;

  if (config_.get().filter_allocate_is_gain_) {
    if (candidate_model_settings_.size() == 1) {
      for (auto &filter_info : filter_infos_) {
        filter_info.model_setting = candidate_model_settings_[0];
      }

      auto min_nseries = static_cast<ID_TYPE>(candidate_model_settings_[0].gpu_ms_per_query / cpu_ms_per_series_);
      spdlog::info("allocator re-assign (single), derived min_nseries = {:d}", min_nseries);

      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

      for (auto &filter_info : filter_infos_) {
        if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_
            || filter_info.node_.get().get_size() < min_nseries) {
          break;
        } else {
          assert(filter_info.model_setting.get().gpu_mem_mb > 0);
          assert(filter_info.model_setting.get().gpu_ms_per_query > 0);
        }

        if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
#ifdef DEBUG
#ifndef DEBUGGED
          spdlog::debug("allocator node {:d} gpu_mem_mb = {:.3f}mb",
                        filter_info.node_.get().get_id(),
                        filter_info.model_setting.get().gpu_mem_mb);
#endif
#endif

          allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
          allocated_filters_count += 1;
        }
      }
    } else {
      // TODO is reassignment possible for multi models?
    }
  } else {
    // default: implant the default model to all leaf nodes
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
        if (filter_info.node_.get().get_size() > config_.get().filter_node_size_threshold_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        }
      }
    }
  }

  spdlog::info("allocator re-assigned {:d} models of {:.1f}/{:.1f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}

RESPONSE dstree::Allocator::set_confidence_from_recall() {
#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("allocator filter_infos_.size() = {:d}",
                filter_infos_.size());
#endif
#endif

  if (!is_recall_calculated_) {
    ID_TYPE num_conformal_examples, num_train_examples;

    if (config_.get().filter_train_num_local_example_ > 0) {
      // contains both global and local examples
      num_train_examples = static_cast<ID_TYPE>(
          config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_);
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else {
      // only contains global examples
      num_train_examples = static_cast<ID_TYPE>(
          config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_);
      ID_TYPE num_valid_examples = config_.get().filter_train_nexample_ - num_train_examples;

      num_conformal_examples = num_valid_examples;
    }

    auto nn_distances = upcite::make_reserved<VALUE_TYPE>(num_conformal_examples);
    auto nn_filter_ids = upcite::make_reserved<ID_TYPE>(num_conformal_examples);

    for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
      nn_distances.push_back(constant::MAX_VALUE);
      nn_filter_ids.push_back(-1);

      for (ID_TYPE filter_i = 0; filter_i < filter_infos_.size(); ++filter_i) {
//    if (filter_infos_[i].node_.get().has_active_filter()) { // nn should be searched from all nodes
        VALUE_TYPE
            local_nn = filter_infos_[filter_i].node_.get().get_filter_nn_distance(num_train_examples + query_i);

        if (local_nn < nn_distances[query_i]) {
          nn_distances[query_i] = local_nn;
          nn_filter_ids[query_i] = filter_i;
        }
      }
    }

#ifdef DEBUG
    //#ifndef DEBUGGED
    spdlog::debug("allocator nn_distances = {:s}",
                  upcite::array2str(nn_distances.data(), num_conformal_examples));
    spdlog::debug("allocator nn_filter_ids = {:s}",
                  upcite::array2str(nn_filter_ids.data(), num_conformal_examples));
    //#endif
#endif

    // two sentry points: recall at small error (recall_at_0_error, 0_error), recall at large error (0.999999, 42)
    // corresponding sentry errors should be added when building the conformal predictors
    validation_recalls_.reserve(num_conformal_examples + 2);

    // recall at validation error intervals
    for (ID_TYPE sorted_error_i = 0; sorted_error_i < num_conformal_examples + 2; ++sorted_error_i) {
      ID_TYPE hit_count = 0;

      for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        std::reference_wrapper<dstree::Node> target_node = filter_infos_[nn_filter_ids[query_i]].node_;

        if (target_node.get().has_active_filter()) {
          VALUE_TYPE abs_error_interval = target_node.get().get_filter_abs_error_interval_by_pos(sorted_error_i);

          VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(num_train_examples + query_i);
          VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(num_train_examples + query_i);

          if (pred_distance - abs_error_interval <= bsf_distance) {
            hit_count += 1;
          }
        } else {
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
        if (filter_info.node_.get().set_filter_abs_error_interval_by_recall(config_.get().filter_conformal_recall_)
            == FAILURE) {
          spdlog::error("allocator failed to get node {:d} conformed at recall {:.3f}",
                        filter_info.node_.get().get_id(),
                        config_.get().filter_conformal_recall_);
        }

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
        if (filter_info.node_.get().set_filter_abs_error_interval_by_pos(last_recall_i) == FAILURE) {
          spdlog::error("allocator failed to get node {:d} conformed with abs {:.3f} at pos {:d}",
                        filter_info.node_.get().get_id(),
                        filter_info.node_.get().get_filter_abs_error_interval_by_pos(last_recall_i),
                        last_recall_i);
        }
      }
    }
  }

  return SUCCESS;
}
