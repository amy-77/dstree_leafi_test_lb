//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <memory>
#include <random>
#include <algorithm>
#include <immintrin.h>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"
#include "eapca.h"
#include "answer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Index::Index(Config &config) :
    config_(config),
    nnode_(0),
    nleaf_(0),
    filter_train_query_ptr_(nullptr) {
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_);

  root_ = std::make_unique<dstree::Node>(config_, *buffer_manager_, 0, nnode_);
  nnode_ += 1, nleaf_ += 1;
}

dstree::Index::~Index() {
  if (filter_train_query_ptr_ != nullptr) {
    std::free(filter_train_query_ptr_);
    filter_train_query_ptr_ = nullptr;
  }
}

RESPONSE dstree::Index::build() {
  while (buffer_manager_->load_batch() == SUCCESS) {
    for (ID_TYPE series_id = 0; series_id < buffer_manager_->load_buffer_size(); ++series_id) {
      insert(series_id);
    }

    if (config_.get().on_disk_) {
      buffer_manager_->flush();
    }
  }

  if (!buffer_manager_->is_fully_loaded()) {
    return FAILURE;
  }

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare>(
      Compare(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  if (config_.get().require_neurofilter_) {
    train();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::insert(ID_TYPE batch_series_id) {
  if (config_.get().is_sketch_provided_) {
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_sketch_ptr(batch_series_id),
        config_.get().sketch_length_,
        config_.get().vertical_split_nsubsegment_)));
  } else {
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_series_ptr(batch_series_id),
        config_.get().series_length_,
        config_.get().vertical_split_nsubsegment_)));
  }
  dstree::EAPCA &series_eapca = buffer_manager_->get_series_eapca(batch_series_id);

  std::reference_wrapper<dstree::Node> target_node = std::ref(*root_);

  while (!target_node.get().is_leaf()) {
    target_node = target_node.get().route(series_eapca);
  }

  if (target_node.get().is_full()) {
    target_node.get().split(*buffer_manager_, nnode_);
    nnode_ += config_.get().node_nchild_, nleaf_ += config_.get().node_nchild_ - 1;

    target_node = target_node.get().route(series_eapca);
  }

  return target_node.get().insert(batch_series_id, series_eapca);
}

RESPONSE dstree::Index::filter_initialize(dstree::Node &node,
                                          ID_TYPE *filter_id) {
  if (node.is_leaf()) {
    node.implant_filter(*filter_id, filter_train_query_tsr_);

    filter_cache_.push(node.get_filter());
    *filter_id += 1;
  } else {
    for (auto child_node : node) {
      filter_initialize(child_node, filter_id);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Index::filter_collect() {
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256),
                                                                   sizeof(VALUE_TYPE) * 8));

  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id) {
    const VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;

    ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
    auto answer = std::make_shared<dstree::Answer>(config_.get().n_nearest_neighbor_, query_id);
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }

    ID_TYPE resident_node_id = resident_node.get().get_id();

    VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, m256_fetch_cache);
    resident_node.get().push_filter_example(answer->get_bsf(), local_nn_distance);

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    if (answer->is_bsf(local_nn_distance)) {
      spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
                   query_id, local_nn_distance, visited_node_counter, visited_series_counter);

      answer->push_bsf(local_nn_distance);
    }

    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), 0));

    // ref: https://stackoverflow.com/questions/9055778/initializing-a-reference-to-member-to-null-in-c
    // TODO better workarounds?
    // WARN undefined behaviour
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *) nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()) {
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();

      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()) {
        if (node_to_visit.get().get_id() != resident_node_id) {
          local_nn_distance = node_to_visit.get().search(series_ptr, m256_fetch_cache);
          node_to_visit.get().push_filter_example(answer->get_bsf(), local_nn_distance);

          visited_node_counter += 1;
          visited_series_counter += node_to_visit.get().get_size();

          if (answer->is_bsf(local_nn_distance)) {
            spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
                         query_id, local_nn_distance, visited_node_counter, visited_series_counter);

            answer->push_bsf(local_nn_distance);
          }
        }
      } else {
        for (auto child_node : node_to_visit.get()) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(series_ptr);

          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }
    }

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger_.get().logger, trivial::info) << boost::format(
          "filter query %d visited %d nodes %d series")
          % query_id
          % visited_node_counter
          % visited_series_counter;
#endif
#endif
  }

  std::free(m256_fetch_cache);
  return SUCCESS;
}

struct SearchCache {
  SearchCache(ID_TYPE thread_id,
              VALUE_TYPE *m256_fetch_cache,
              dstree::Answer *answer,
              pthread_mutex_t *answer_mutex,
              std::reference_wrapper<std::priority_queue<dstree::NODE_DISTNCE,
                                                         std::vector<dstree::NODE_DISTNCE>,
                                                         dstree::Compare>> leaf_min_heap,
              pthread_mutex_t *leaf_pq_mutex,
              ID_TYPE *visited_node_counter,
              ID_TYPE *visited_series_counter,
              pthread_mutex_t *log_mutex) :
      thread_id_(thread_id),
      query_id_(-1),
      query_series_ptr_(nullptr),
      m256_fetch_cache_(m256_fetch_cache),
      answer_(answer),
      answer_mutex_(answer_mutex),
      leaf_min_heap_(leaf_min_heap),
      leaf_pq_mutex_(leaf_pq_mutex),
      visited_node_counter_(visited_node_counter),
      visited_series_counter_(visited_series_counter),
      log_mutex_(log_mutex) {}

  ID_TYPE thread_id_;

  ID_TYPE query_id_;
  VALUE_TYPE *query_series_ptr_;

  VALUE_TYPE *m256_fetch_cache_;

  dstree::Answer *answer_;
  pthread_mutex_t *answer_mutex_;

  std::reference_wrapper<std::priority_queue<dstree::NODE_DISTNCE, std::vector<dstree::NODE_DISTNCE>, dstree::Compare>>
      leaf_min_heap_;
  pthread_mutex_t *leaf_pq_mutex_;

  ID_TYPE *visited_node_counter_;
  ID_TYPE *visited_series_counter_;

  pthread_mutex_t *log_mutex_;
};

void search_thread_F(const SearchCache &search_cache) {
  // aligned_alloc within thread might cause a "corrupted size vs. prev_size" glibc error
  // https://stackoverflow.com/questions/49628615/understanding-corrupted-size-vs-prev-size-glibc-error
//  auto m256_fetch_cache = std::unique_ptr<VALUE_TYPE>(static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8)));

  // WARN undefined behaviour
  std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *) nullptr);
  VALUE_TYPE node2visit_lbdistance;
  VALUE_TYPE local_nn_distance;

  pthread_mutex_lock(search_cache.answer_mutex_);
  VALUE_TYPE local_bsf = search_cache.answer_->get_bsf();
  pthread_mutex_unlock(search_cache.answer_mutex_);

#ifdef DEBUG
#ifndef DEBUGGED
  pthread_mutex_lock(search_cache.leaf_pq_lock_);
  spdlog::info("filter query {:d} thread {:d} initial bsf {:.3f} before {:d} nodes",
               search_cache.query_id_,
               search_cache.thread_id_,
               local_bsf,
               search_cache.leaf_min_heap_.get().size());
  pthread_mutex_unlock(search_cache.leaf_pq_lock_);
#endif
#endif

  while (true) {
//    spdlog::info("filter query {:d} thread {:d} before wrlock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    pthread_mutex_lock(search_cache.leaf_pq_mutex_);

//    spdlog::info("filter query {:d} thread {:d} after wrlock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    if (search_cache.leaf_min_heap_.get().empty()) {
//      spdlog::info("filter query {:d} thread {:d} before empty unlock",
//                  search_cache.query_id_,
//                  search_cache.thread_id_);

      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);

//      spdlog::info("filter query {:d} thread {:d} after empty unlock",
//                  search_cache.query_id_,
//                  search_cache.thread_id_);

      break;
    } else {
      std::tie(node_to_visit, node2visit_lbdistance) = search_cache.leaf_min_heap_.get().top();
      search_cache.leaf_min_heap_.get().pop();

//      spdlog::info("filter query {:d} thread {:d} before pop unlock",
//                  search_cache.query_id_,
//                  search_cache.thread_id_);

      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);

//      spdlog::info("filter query {:d} thread {:d} after pop unlock",
//                  search_cache.query_id_,
//                  search_cache.thread_id_);
#ifdef DEBUG
#ifndef DEBUGGED
      pthread_mutex_lock(search_cache.leaf_pq_mutex_);
      spdlog::info("filter query {:d} thread {:d} visit node {:d} before {:d} nodes",
                  search_cache.query_id_,
                  search_cache.thread_id_,
                  node_to_visit.get().get_id(),
                  search_cache.leaf_min_heap_.get().size());
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
#endif
#endif
    }

    if (node_to_visit.get().has_filter()) {
      local_nn_distance = node_to_visit.get().search(
          search_cache.query_series_ptr_, search_cache.m256_fetch_cache_);

//      spdlog::info("filter query {:d} thread {:d} local_nn_distance = {:.3f}",
//                   search_cache.query_id_,
//                   search_cache.thread_id_,
//                   local_nn_distance);

      node_to_visit.get().push_filter_example(local_bsf, local_nn_distance);
    } else {
      local_nn_distance = node_to_visit.get().search(
          search_cache.query_series_ptr_, search_cache.m256_fetch_cache_, local_bsf);
    }

//    spdlog::info("filter query {:d} thread {:d} before mutex_lock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    pthread_mutex_lock(search_cache.log_mutex_);

//    spdlog::info("filter query {:d} thread {:d} after mutex_lock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    *search_cache.visited_node_counter_ += 1;
    *search_cache.visited_series_counter_ += node_to_visit.get().get_size();

//    spdlog::info("filter query {:d} thread {:d} before mutex_unlock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    pthread_mutex_unlock(search_cache.log_mutex_);

//    spdlog::info("filter query {:d} thread {:d} after mutex_unlock",
//                search_cache.query_id_,
//                search_cache.thread_id_);

    if (local_nn_distance < local_bsf) {
      pthread_mutex_lock(search_cache.answer_mutex_);
      search_cache.answer_->push_bsf(local_nn_distance);

      local_bsf = search_cache.answer_->get_bsf();
      pthread_mutex_unlock(search_cache.answer_mutex_);

#ifdef DEBUG
//#ifndef DEBUGGED
      if (local_bsf > local_nn_distance - constant::EPSILON) {
        spdlog::info("filter query {:d} thread {:d} update bsf {:.3f} after node {:d} series {:d}",
                     search_cache.query_id_,
                     search_cache.thread_id_,
                     local_nn_distance,
                     *search_cache.visited_node_counter_,
                     *search_cache.visited_series_counter_);
      }
//#endif
#endif
    }
  }
}

RESPONSE dstree::Index::filter_collect_mthread() {
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(
      sizeof(__m256), 8 * config_.get().filter_collect_nthread_));

  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;

  auto answer = std::make_unique<dstree::Answer>(config_.get().n_nearest_neighbor_, -1);

  std::unique_ptr<pthread_mutex_t> answer_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> leaf_pq_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> log_mutex = std::make_unique<pthread_mutex_t>();

  pthread_mutex_init(answer_mutex.get(), nullptr);
  pthread_mutex_init(leaf_pq_mutex.get(), nullptr);
  pthread_mutex_init(log_mutex.get(), nullptr);

  std::vector<SearchCache> search_caches;
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id) {
    search_caches.emplace_back(thread_id,
                               m256_fetch_cache + 8 * thread_id,
                               answer.get(),
                               answer_mutex.get(),
                               std::ref(leaf_min_heap_),
                               leaf_pq_mutex.get(),
                               &visited_node_counter,
                               &visited_series_counter,
                               log_mutex.get());
  }

  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id) {
    VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;

    visited_node_counter = 0;
    visited_series_counter = 0;
    answer->reset(query_id);

    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id) {
      search_caches[thread_id].query_id_ = query_id;
      search_caches[thread_id].query_series_ptr_ = series_ptr;
    }

    VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, m256_fetch_cache);
    resident_node.get().push_filter_example(answer->get_bsf(), local_nn_distance);

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    if (answer->is_bsf(local_nn_distance)) {
      spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
                   query_id, local_nn_distance, visited_node_counter, visited_series_counter);

      answer->push_bsf(local_nn_distance);
    }

    assert(node_stack.empty() && leaf_min_heap_.empty());
    node_stack.push(std::ref(*root_));

    while (!node_stack.empty()) {
      std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
      node_stack.pop();

      if (node_to_visit.get().is_leaf()) {
        if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
          leaf_min_heap_.push(std::make_tuple(node_to_visit,
                                              node_to_visit.get().cal_lower_bound_EDsquare(series_ptr)));
        }
      } else {
        for (auto child_node : node_to_visit.get()) {
          node_stack.push(child_node);
        }
      }
    }

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::info("filter query {:d} enqueued {:d} nodes",
                 query_id, leaf_min_heap_.size());
#endif
#endif

    std::vector<std::thread> threads;

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id) {
      threads.emplace_back(search_thread_F, search_caches[thread_id]);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id) {
      threads[thread_id].join();
    }

#ifdef DEBUG
    //#ifndef DEBUGGED
    spdlog::info("filter query {:d} visited {:d} nodes {:d} series",
                 query_id, visited_node_counter, visited_series_counter);
    //#endif
#endif
  }

  std::free(m256_fetch_cache);
  return SUCCESS;
}

RESPONSE dstree::Index::filter_train() {
  while (!filter_cache_.empty()) {
    std::reference_wrapper<Filter> filter = filter_cache_.top();
    filter_cache_.pop();

    filter.get().train();
  }

  return SUCCESS;
}

struct TrainCache {
  TrainCache(ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache,
             pthread_mutex_t *filter_cache_mutex) :
      thread_id_(thread_id),
      stream_(stream),
      filter_cache_(filter_cache),
      filter_cache_mutex_(filter_cache_mutex) {}
  ~TrainCache() = default;

  ID_TYPE thread_id_;

  at::cuda::CUDAStream stream_;

  std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache_;
  pthread_mutex_t *filter_cache_mutex_;
};

void train_thread_F(TrainCache &train_cache) {
  at::cuda::setCurrentCUDAStream(train_cache.stream_);
  at::cuda::CUDAStreamGuard guard(train_cache.stream_); // compiles with libtorch-gpu

  while (true) {
    pthread_mutex_lock(train_cache.filter_cache_mutex_);
    if (train_cache.filter_cache_.empty()) {
      pthread_mutex_unlock(train_cache.filter_cache_mutex_);

      break;
    } else {
      std::reference_wrapper<dstree::Filter> filter = train_cache.filter_cache_.top();
      train_cache.filter_cache_.pop();

      pthread_mutex_unlock(train_cache.filter_cache_mutex_);

      filter.get().train();
    }
  }
}

RESPONSE dstree::Index::filter_train_mthread() {
  // TODO enable multithread train on CPU
  assert(config_.get().filter_train_is_gpu_);
  assert(torch::cuda::is_available());

  std::stack<std::reference_wrapper<dstree::Filter>> filters;
  std::unique_ptr<pthread_mutex_t> filter_stack_mutex = std::make_unique<pthread_mutex_t>();

  std::stack<std::reference_wrapper<dstree::Node>> node_stack;
  node_stack.push(std::ref(*root_));

  while (!node_stack.empty()) {
    std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
    node_stack.pop();

    if (node_to_visit.get().is_leaf()) {
      if (node_to_visit.get().has_filter()) {
        filters.push(node_to_visit.get().get_filter());
      }
    } else {
      for (auto child_node : node_to_visit.get()) {
        node_stack.push(child_node);
      }
    }
  }

#ifdef DEBUG
  //#ifndef DEBUGGED
  spdlog::debug("filter en-stack {:d} filters", filters.size());
  //#endif
#endif

  std::vector<std::unique_ptr<TrainCache>> train_caches;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    spdlog::info("thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
                 thread_id,
                 static_cast<ID_TYPE>(new_stream.id()),
                 static_cast<ID_TYPE>(new_stream.query()),
                 static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu

    train_caches.emplace_back(std::make_unique<TrainCache>(thread_id,
                                                           std::move(new_stream),
                                                           std::ref(filters),
                                                           filter_stack_mutex.get()));
  }

  std::vector<std::thread> threads;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads.emplace_back(train_thread_F, std::ref(*train_caches[thread_id]));
  }

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads[thread_id].join();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::train() {
  auto query_nbytes = static_cast<ID_TYPE>(
      sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().filter_train_nexample_;
  filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

  if (!fs::exists(config_.get().filter_query_filepath_)) {
    if (config_.get().filter_query_filepath_ != "") {
      spdlog::error("filter train query filepath {:s} does not exist", config_.get().filter_query_filepath_);
    }

    auto sampled_ids = static_cast<ID_TYPE *>(malloc(sizeof(ID_TYPE) * config_.get().db_nseries_));
    for (ID_TYPE i = 0; i < config_.get().db_nseries_; ++i) {
      sampled_ids[i] = i;
    }

    std::random_device rd;
    std::mt19937 rng{rd()};
    std::uniform_int_distribution<ID_TYPE> rand_uniform(0, constant::MAX_ID);
    std::shuffle(sampled_ids, sampled_ids + config_.get().db_nseries_, rng);

    std::string filter_query_id_filepath =
        config_.get().index_persist_folderpath_ + config_.get().filter_query_id_filename_;
    std::ofstream id_fout(filter_query_id_filepath, std::ios::binary | std::ios_base::app);
    id_fout.write(reinterpret_cast<char *>(sampled_ids), sizeof(ID_TYPE) * config_.get().filter_train_nexample_);
    id_fout.close();

    std::ifstream db_fin(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    ID_TYPE series_nbytes = sizeof(VALUE_TYPE) * config_.get().series_length_;

    for (ID_TYPE i = 0; i < config_.get().filter_train_nexample_; ++i) {
      VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * i;

      ID_TYPE series_bytes_offset = series_nbytes * sampled_ids[i];
      db_fin.seekg(series_bytes_offset);
      db_fin.read(reinterpret_cast<char *>(series_ptr), series_nbytes);

      VALUE_TYPE mean = 0, std_dev = 0;
      for (ID_TYPE j = 0; j < config_.get().series_length_; ++j) {
        VALUE_TYPE f1 = static_cast<VALUE_TYPE>(rand_uniform(rng) % 999 + 1) / 1000.0f;
        VALUE_TYPE c = static_cast<VALUE_TYPE>(sqrt(-2.0 * log(f1)));

        VALUE_TYPE f2 = static_cast<VALUE_TYPE>(rand_uniform(rng) % 1000) / 1000.0f;
        VALUE_TYPE b = 2 * constant::PI_APPROX_7 * f2;

        VALUE_TYPE noise = c * cos(b) * sqrt(config_.get().filter_query_noise_level_);

        series_ptr[j] += noise;
        mean += series_ptr[j];
      }

      mean /= config_.get().series_length_;

      for (ID_TYPE j = 0; j < config_.get().series_length_; ++j) {
        std_dev += (series_ptr[j] - mean) * (series_ptr[j] - mean);
      }

      std_dev = sqrt(std_dev / config_.get().series_length_);

      for (ID_TYPE j = 0; j < config_.get().series_length_; ++j) {
        series_ptr[j] = (series_ptr[j] - mean) / std_dev;
      }
    }

    std::string filter_query_filepath = config_.get().index_persist_folderpath_ + config_.get().filter_query_filename_;
    std::ofstream query_fout(filter_query_filepath, std::ios::binary | std::ios_base::app);
    query_fout.write(reinterpret_cast<char *>(filter_train_query_ptr_),
                     series_nbytes * config_.get().filter_train_nexample_);
    query_fout.close();

    spdlog::info("generated {:d} filter train queries", config_.get().filter_train_nexample_);
  } else {
    std::ifstream query_fin(config_.get().filter_query_filepath_, std::ios::in | std::ios::binary);
    if (!query_fin.good()) {
      spdlog::error("filter train query filepath {:s} cannot open", config_.get().filter_query_filepath_);
    }

    query_fin.read(reinterpret_cast<char *>(filter_train_query_ptr_), query_nbytes);

    if (query_fin.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().filter_query_filepath_);

      std::free(filter_train_query_ptr_);
      filter_train_query_ptr_ = nullptr;
    }
  }

  if (config_.get().filter_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                              static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  filter_train_query_tsr_ = torch::from_blob(filter_train_query_ptr_,
                                             {config_.get().filter_train_nexample_, config_.get().series_length_},
                                             torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  filter_train_query_tsr_ = filter_train_query_tsr_.to(*device_);

  ID_TYPE filter_id = 0;
  filter_initialize(*root_, &filter_id);

  spdlog::info("initialized {:d} filters", filter_id);

  if (config_.get().filter_train_is_mthread_) {
    filter_collect_mthread();
  } else {
    filter_collect();
  }

  if (config_.get().filter_train_is_mthread_) {
    filter_train_mthread();
  } else {
    filter_train();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::load() {
  // TODO

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare>(
      Compare(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  return FAILURE;
}

RESPONSE dstree::Index::dump() const {
  ID_TYPE ofs_buf_size = sizeof(ID_TYPE) * config_.get().series_length_ * 2; // 2x expanded for safety
  void *ofs_buf = std::malloc(ofs_buf_size);

  root_->dump(ofs_buf);

  std::free(ofs_buf);
  return SUCCESS;
}

RESPONSE dstree::Index::search() {
  if (!fs::exists(config_.get().query_filepath_)) {
    spdlog::error("query filepath {:s} does not exist", config_.get().query_filepath_);

    return FAILURE;
  }

  std::ifstream query_fin(config_.get().query_filepath_, std::ios::in | std::ios::binary);
  if (!query_fin.good()) {
    spdlog::error("query filepath {:s} cannot open", config_.get().query_filepath_);

    return FAILURE;
  }

  auto query_nbytes =
      static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().query_nseries_;
  auto query_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_nbytes));

  query_fin.read(reinterpret_cast<char *>(query_buffer), query_nbytes);

  if (query_fin.fail()) {
    spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);

    return FAILURE;
  }

  VALUE_TYPE *query_sketch_buffer = nullptr;
  if (config_.get().is_sketch_provided_) {
    if (!fs::exists(config_.get().query_sketch_filepath_)) {
      spdlog::error("query sketch filepath {:s} does not exist", config_.get().query_sketch_filepath_);

      return FAILURE;
    }

    std::ifstream query_sketch_fin(config_.get().query_sketch_filepath_, std::ios::in | std::ios::binary);
    if (!query_fin.good()) {
      spdlog::error("query sketch filepath {:s} cannot open", config_.get().query_sketch_filepath_);

      return FAILURE;
    }

    auto query_sketch_nbytes =
        static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * config_.get().query_nseries_;
    query_sketch_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_sketch_nbytes));

    query_sketch_fin.read(reinterpret_cast<char *>(query_sketch_buffer), query_sketch_nbytes);

    if (query_sketch_fin.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);

      return FAILURE;
    }
  }

  VALUE_TYPE *series_ptr, *sketch_ptr = nullptr;
  for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
    series_ptr = query_buffer + config_.get().series_length_ * query_id;

    if (config_.get().is_sketch_provided_) {
      sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
    }

    search(query_id, series_ptr, sketch_ptr);
  }

  return SUCCESS;
}

RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr) {
  VALUE_TYPE *route_ptr = series_ptr;
  if (config_.get().is_sketch_provided_) {
    route_ptr = sketch_ptr;
  }

  auto answer = std::make_shared<dstree::Answer>(config_.get().n_nearest_neighbor_, query_id);

  if (config_.get().require_neurofilter_) {
    filter_query_tsr_ = torch::from_blob(series_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    filter_query_tsr_ = filter_query_tsr_.to(*device_);
  }

  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

  while (!resident_node.get().is_leaf()) {
    resident_node = resident_node.get().route(route_ptr);
  }

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;

  resident_node.get().search(series_ptr, *answer, visited_node_counter, visited_series_counter);

  if (config_.get().is_exact_search_) {
    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));

    // WARN undefined behaviour
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *) nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()) {
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger_.get().logger, trivial::debug) << boost::format(
            "query %d node_id %d leaf_min_heap_.size %d node2visit_lbdistance %.3f bsf %.3f")
            % answer->query_id_
            % node_to_visit->get_id()
            % leaf_min_heap_.size()
            % node2visit_lbdistance
            % answer->get_bsf();
#endif
#endif

      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()) {
        if (visited_node_counter < config_.get().search_max_nnode_ &&
            visited_series_counter < config_.get().search_max_nseries_) {
          if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
            if (config_.get().examine_ground_truth_ || answer->is_bsf(node2visit_lbdistance)) {
              if (node_to_visit.get().has_filter()) {
                VALUE_TYPE predicted_nn_distance = node_to_visit.get().filter_infer(filter_query_tsr_);

#ifdef DEBUG
//#ifndef DEBUGGED
                spdlog::debug("query {:d} node_id {:d} d_pred_sq {:.3f} bsf {:.3f}",
                              answer->query_id_,
                              node_to_visit.get().get_id(),
                              predicted_nn_distance,
                              answer->get_bsf());
//#endif
#endif

                if (predicted_nn_distance < answer->get_bsf()) {
                  node_to_visit.get().search(series_ptr, *answer, visited_node_counter, visited_series_counter);
                } else {
                  nfpruned_node_counter += 1;
                  nfpruned_series_counter += node_to_visit.get().get_size();
                }
              } else {
                node_to_visit.get().search(series_ptr, *answer, visited_node_counter, visited_series_counter);
              }
            }
          }
        }
      } else {
        for (auto child_node : node_to_visit.get()) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);

          if (config_.get().examine_ground_truth_ || answer->is_bsf(child_lower_bound_EDsquare)) {
            leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
          }
        }
      }
    }
  }

  spdlog::info("query {:d} visited {:d} nodes {:d} series",
               query_id, visited_node_counter, visited_series_counter);

  if (config_.get().require_neurofilter_) {
    spdlog::info("query {:d} neurofilters pruned {:d} nodes {:d} series",
                 query_id, nfpruned_node_counter, nfpruned_series_counter);
  }

  ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
  while (!answer->empty()) {
    spdlog::info("query {:d} nn {:d} = {:.3f}",
                 query_id, nnn_to_return, answer->pop_bsf());

    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0) {
    return FAILURE;
  }

  return SUCCESS;
}
