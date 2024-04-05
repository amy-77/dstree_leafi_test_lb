//
// Created by Qitong Wang on 2024/3/26.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#include "query_synthesizer.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <immintrin.h>

#include "stat.h"
#include "sort.h"

namespace dstree = upcite::dstree;

dstree::Synthesizer::Synthesizer(dstree::Config &config,
                                 ID_TYPE num_leaves) :
    config_(config),
    num_leaves_(num_leaves) {

  if (num_leaves_ > 0) {
    leaves_.reserve(num_leaves_);
    accumulated_leaf_sizes_.reserve(num_leaves_);
  }
}

RESPONSE dstree::Synthesizer::push_node(dstree::Node &leaf_node) {
  leaves_.emplace_back(leaf_node);
  accumulated_leaf_sizes_.push_back(accumulated_leaf_sizes_.back() + leaf_node.get_size());

  return SUCCESS;
}

RESPONSE dstree::Synthesizer::generate_global_data(VALUE_TYPE *generated_queries) {
  ID_TYPE num_global_examples = config_.get().filter_train_num_global_example_;
  ID_TYPE num_series_within_filters = accumulated_leaf_sizes_.back();

  // provided by ChatGPT
  const gsl_rng_type *T;
  gsl_rng *r;
  // Create a generator chosen by the environment variable GSL_RNG_TYPE
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, (unsigned long) time(nullptr));

  for (ID_TYPE query_i = 0; query_i < num_global_examples; ++query_i) {
    auto random_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, num_series_within_filters));
    ID_TYPE leaf_i = upcite::bSearchFloorID(random_i, accumulated_leaf_sizes_.data(),
                                            0, accumulated_leaf_sizes_.size() - 1);
    auto series_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, leaves_[leaf_i].get().get_size()));

    VALUE_TYPE const *series_ptr = leaves_[leaf_i].get().get_series_ptr_by_id(series_i);

    VALUE_TYPE noise_level = config_.get().filter_query_min_noise_ + gsl_rng_uniform(r) * (
        config_.get().filter_query_max_noise_ - config_.get().filter_query_min_noise_);

    VALUE_TYPE *series_to_generate = generated_queries + config_.get().series_length_ * query_i;
    for (ID_TYPE value_i = 0; value_i < config_.get().series_length_; ++value_i) {
      series_to_generate[value_i] = series_ptr[value_i] + static_cast<VALUE_TYPE>(gsl_ran_gaussian(r, noise_level));
    }

    RESPONSE return_code = upcite::znormalize(series_to_generate, config_.get().series_length_);
    if (return_code == FAILURE) {
      spdlog::error("node {:d} failed to znorm series {:d} +noise {:.3f}; regenerate",
                    leaves_[leaf_i].get().get_id(), series_i, noise_level);
      query_i -= 1;
    }
  }

  gsl_rng_free(r);

  return SUCCESS;
}

struct LocalGenerationCache {
  LocalGenerationCache(dstree::Config &config,
                       ID_TYPE thread_id,
                       std::vector<std::reference_wrapper<dstree::Node>> &leaves,
                       ID_TYPE *leaf_i,
                       pthread_mutex_t *leaves_mutex) :
      config_(config),
      thread_id_(thread_id),
      leaves_(leaves),
      leaf_i_(leaf_i),
      leaves_mutex_(leaves_mutex) {}

  ~LocalGenerationCache() = default;

  std::reference_wrapper<dstree::Config> config_;
  ID_TYPE thread_id_;

  std::vector<std::reference_wrapper<dstree::Node>> leaves_;
  ID_TYPE *leaf_i_;
  pthread_mutex_t *leaves_mutex_;
};

void generation_thread_F(LocalGenerationCache &generation_cache) {
  ID_TYPE num_local_queries = generation_cache.config_.get().filter_train_num_local_example_;
  ID_TYPE series_length = generation_cache.config_.get().series_length_;

  VALUE_TYPE max_noise = generation_cache.config_.get().filter_query_max_noise_;
  VALUE_TYPE min_noise = generation_cache.config_.get().filter_query_min_noise_;

  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
  auto generated_series_ptr = static_cast<VALUE_TYPE *>(aligned_alloc(
      sizeof(__m256), sizeof(VALUE_TYPE) * series_length));

  // provided by ChatGPT
  const gsl_rng_type *T;
  gsl_rng *r;
  // Create a generator chosen by the environment variable GSL_RNG_TYPE
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc(T);
  gsl_rng_set(r, (unsigned long) time(nullptr));

  ID_TYPE num_leaves = generation_cache.leaves_.size();

  pthread_mutex_lock(generation_cache.leaves_mutex_);
  ID_TYPE local_leaf_i = *generation_cache.leaf_i_;
  *generation_cache.leaf_i_ += 1;
  pthread_mutex_unlock(generation_cache.leaves_mutex_);

  while (local_leaf_i < num_leaves) {
    dstree::Node &current_node = generation_cache.leaves_[local_leaf_i];

    VALUE_TYPE mean, std;
    std::tie(mean, std) = current_node.get_filter_global_lnn_mean_std();
    VALUE_TYPE max_legal_lnn_distance = mean - std;

    if (max_legal_lnn_distance <= 0) {
      spdlog::error("thread {:d} node {:d} broken global nn dist stats, mean = {:.3f} std = {:.3f}",
                    generation_cache.thread_id_, current_node.get_id(), mean, std);
      spdlog::shutdown();
      exit(FAILURE);
    }

    for (ID_TYPE query_i = 0; query_i < num_local_queries; ++query_i) {
      auto series_i = static_cast<ID_TYPE>(gsl_rng_uniform_int(r, current_node.get_size()));
      VALUE_TYPE const *series_ptr = current_node.get_series_ptr_by_id(series_i);

      VALUE_TYPE noise_level = min_noise + gsl_rng_uniform(r) * (max_noise - min_noise);

      for (ID_TYPE value_i = 0; value_i < series_length; ++value_i) {
        generated_series_ptr[value_i] = series_ptr[value_i] + static_cast<VALUE_TYPE>(gsl_ran_gaussian(r, noise_level));
      }

      RESPONSE return_code = upcite::znormalize(generated_series_ptr, series_length);
      if (return_code == FAILURE) {
        spdlog::error("thread {:d} node {:d} failed to znorm series {:d} +noise {:.3f}; regenerate",
                      generation_cache.thread_id_, current_node.get_id(), series_i, noise_level);
        query_i -= 1;
      } else {
        // TODO check and prune if the generated series falling in the same leaf node
        // does it work for the EAPCA envelop of dstree?

        VALUE_TYPE local_nn_distance = current_node.search(generated_series_ptr, m256_fetch_cache);

        if (local_nn_distance > max_legal_lnn_distance) {
          spdlog::error(
              "thread {:d} node {:d} series {:d} +noise {:.3f} escaped the local neighbourhood ({:.3f} > {:.3f}); regenerate",
              generation_cache.thread_id_,
              current_node.get_id(),
              series_i,
              noise_level,
              local_nn_distance,
              max_legal_lnn_distance);

          query_i -= 1;
        } else {
          current_node.push_local_example(generated_series_ptr, local_nn_distance);

          spdlog::info("thread {:d} node {:d} series {:d} +noise {:.3f}, lnn = {:.3f} <= {:.3f}",
                       generation_cache.thread_id_, current_node.get_id(), series_i, noise_level,
                       local_nn_distance, max_legal_lnn_distance);
        }
      }
    }

//    current_node.dump_local_example();

    pthread_mutex_lock(generation_cache.leaves_mutex_);
    local_leaf_i = *generation_cache.leaf_i_;
    *generation_cache.leaf_i_ += 1;
    pthread_mutex_unlock(generation_cache.leaves_mutex_);
  }

  gsl_rng_free(r);
  free(m256_fetch_cache);
  free(generated_series_ptr);
}

RESPONSE dstree::Synthesizer::generate_local_data() {
  if (!config_.get().filter_train_is_mthread_) {
    spdlog::error("single-thread query generation not implemented");
    spdlog::shutdown();
    exit(FAILURE);
  }

  std::vector<std::unique_ptr<LocalGenerationCache>> generation_caches;
  ID_TYPE leaf_i = 0;
  std::unique_ptr<pthread_mutex_t> leaves_mutex = std::make_unique<pthread_mutex_t>();

  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    generation_caches.emplace_back(std::make_unique<LocalGenerationCache>(
        config_, thread_i, std::ref(leaves_), &leaf_i, leaves_mutex.get()));
  }

  std::vector<std::thread> threads;

  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    threads.emplace_back(generation_thread_F, std::ref(*generation_caches[thread_i]));
  }

  for (ID_TYPE thread_i = 0; thread_i < config_.get().filter_train_nthread_; ++thread_i) {
    threads[thread_i].join();
  }

  return SUCCESS;
}
