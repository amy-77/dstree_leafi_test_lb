//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <memory>
#include <immintrin.h>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "common.h"
#include "eapca.h"
#include "answer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

dstree::Index::Index(std::shared_ptr<Config> config, std::shared_ptr<upcite::Logger> logger) :
    config_(std::move(config)),
    logger_(std::move(logger)),
    nnode_(0),
    nleaf_(0),
    nf_train_query_ptr_(nullptr) {
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_, logger_);

  root_ = std::make_shared<dstree::Node>(config_, logger_, buffer_manager_, 0, nnode_);
  nnode_ += 1, nleaf_ += 1;
}

dstree::Index::~Index() {
  if (nf_train_query_ptr_ != nullptr) {
    std::free(nf_train_query_ptr_);
    nf_train_query_ptr_ = nullptr;
  }
}

RESPONSE dstree::Index::build() {
  while (buffer_manager_->load_batch() == SUCCESS) {
    for (ID_TYPE series_id = 0; series_id < buffer_manager_->load_buffer_size(); ++series_id) {
      insert(series_id);
    }

    if (config_->on_disk_) {
      buffer_manager_->flush();
    }
  }

  if (!buffer_manager_->is_fully_loaded()) {
    return FAILURE;
  }

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare>(
      Compare(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  if (config_->require_neurofilter_) {
    train();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::insert(ID_TYPE batch_series_id) {
  buffer_manager_->batch_eapca_.push_back(std::make_shared<dstree::EAPCA>(
      buffer_manager_->get_series_ptr(batch_series_id),
      config_->series_length_,
      config_->vertical_split_nsubsegment_));
  std::shared_ptr<dstree::EAPCA> series_eapca = buffer_manager_->batch_eapca_[batch_series_id];

  std::shared_ptr<Node> target_node = root_;

  while (!target_node->is_leaf()) {
    target_node = target_node->route(series_eapca);
  }

  if (target_node->is_full()) {
    target_node->split(config_, buffer_manager_, nnode_);
    nnode_ += config_->node_nchild_, nleaf_ += config_->node_nchild_ - 1;

    target_node = target_node->route(series_eapca);
  }

  return target_node->insert(batch_series_id, series_eapca);
}

RESPONSE dstree::Index::nf_initialize(std::shared_ptr<dstree::Node> &node,
                                      ID_TYPE *filter_id) {
  if (node->is_leaf()) {
    node->neurofilter_ = std::make_unique<dstree::Filter>(logger_, config_, *filter_id, nf_train_query_tsr_);

    filter_cache_.push(std::ref(*node->neurofilter_));
    *filter_id += 1;
  } else {
    for (ID_TYPE child_id = 0; child_id < config_->node_nchild_; ++child_id) {
      nf_initialize(node->children_[child_id], filter_id);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Index::nf_collect() {
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256),
                                                                   sizeof(VALUE_TYPE) * 8));

  for (ID_TYPE query_id = 0; query_id < config_->nf_train_nexample_; ++query_id) {
    const VALUE_TYPE *series_ptr = nf_train_query_ptr_ + config_->series_length_ * query_id;

    ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
    auto answer = std::make_shared<dstree::Answer>(config_->n_nearest_neighbor_, query_id);
    std::shared_ptr<dstree::Node> resident_node = root_;

    while (!resident_node->is_leaf()) {
      resident_node = resident_node->route(series_ptr);
    }

    ID_TYPE resident_node_id = resident_node->get_id();

    VALUE_TYPE local_nn_distance = resident_node->search(series_ptr, m256_fetch_cache);
    resident_node->neurofilter_->push_example(answer->get_bsf(), local_nn_distance);

    visited_node_counter += 1;
    visited_series_counter += resident_node->get_size();

    if (answer->is_bsf(local_nn_distance)) {
      MALAT_LOG(logger_->logger, trivial::info) << boost::format(
            "nf query %d update bsf %f after node %d series %s")
            % query_id
            % local_nn_distance
            % visited_node_counter
            % visited_series_counter;

      answer->push_bsf(local_nn_distance);
    }

    leaf_min_heap_.push(std::make_tuple(root_, 0));

    std::shared_ptr<dstree::Node> node_to_visit;
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()) {
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();

      leaf_min_heap_.pop();

      if (node_to_visit->is_leaf()) {
        if (node_to_visit->get_id() != resident_node_id) {
          local_nn_distance = node_to_visit->search(series_ptr, m256_fetch_cache);
          node_to_visit->neurofilter_->push_example(answer->get_bsf(), local_nn_distance);

          visited_node_counter += 1;
          visited_series_counter += node_to_visit->get_size();

          if (answer->is_bsf(local_nn_distance)) {
            MALAT_LOG(logger_->logger, trivial::info) << boost::format(
                  "nf query %d update bsf %f after node %d series %s")
                  % query_id
                  % local_nn_distance
                  % visited_node_counter
                  % visited_series_counter;

            answer->push_bsf(local_nn_distance);
          }
        }
      } else {
        for (const auto &child_node : node_to_visit->children_) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node->cal_lower_bound_EDsquare(series_ptr);

          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }
    }

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "nf query %d visited %d nodes %d series")
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
              upcite::Logger &logger,
              dstree::Config &config,
              dstree::Answer &answer,
              pthread_rwlock_t *answer_lock,
              std::priority_queue<dstree::NODE_DISTNCE,
                                  std::vector<dstree::NODE_DISTNCE>,
                                  dstree::Compare> &leaf_min_heap,
              pthread_rwlock_t *leaf_pq_lock,
              ID_TYPE *visited_node_counter,
              ID_TYPE *visited_series_counter,
              pthread_mutex_t *log_mutex) :
      thread_id_(thread_id),
      logger_(logger),
      config_(config),
      query_id_(-1),
      query_series_ptr_(nullptr),
      answer_(answer),
      answer_lock_(answer_lock),
      leaf_min_heap_(leaf_min_heap),
      leaf_pq_lock_(leaf_pq_lock),
      visited_node_counter_(visited_node_counter),
      visited_series_counter_(visited_node_counter),
      log_mutex_(log_mutex),
      m256_fetch_cache_(std::unique_ptr<VALUE_TYPE>(static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8)))) {}
  ~SearchCache() = default;

  ID_TYPE thread_id_;

  upcite::Logger &logger_;
  dstree::Config &config_;

  ID_TYPE query_id_;
  VALUE_TYPE *query_series_ptr_;

  dstree::Answer &answer_;
  pthread_rwlock_t *answer_lock_;

  std::priority_queue<dstree::NODE_DISTNCE, std::vector<dstree::NODE_DISTNCE>, dstree::Compare> &leaf_min_heap_;
  pthread_rwlock_t *leaf_pq_lock_;

  std::unique_ptr<VALUE_TYPE> m256_fetch_cache_;

  ID_TYPE *visited_node_counter_;
  ID_TYPE *visited_series_counter_;

  pthread_mutex_t *log_mutex_;
};

void search_thread_F(SearchCache &search_cache) {
  std::shared_ptr<dstree::Node> node_to_visit;
  VALUE_TYPE node2visit_lbdistance;

  VALUE_TYPE local_bsf = search_cache.answer_.get_bsf();
  VALUE_TYPE local_nn_distance;

  while (true) {
    pthread_rwlock_wrlock(search_cache.leaf_pq_lock_);
    if (search_cache.leaf_min_heap_.empty()) {
      pthread_rwlock_unlock(search_cache.leaf_pq_lock_);

      break;
    } else {
      std::tie(node_to_visit, node2visit_lbdistance) = search_cache.leaf_min_heap_.top();
      search_cache.leaf_min_heap_.pop();

      pthread_rwlock_unlock(search_cache.leaf_pq_lock_);
    }

    if (node_to_visit->neurofilter_ == nullptr) {
      local_nn_distance = node_to_visit->search(
          search_cache.query_series_ptr_, search_cache.m256_fetch_cache_.get(), local_bsf);
    } else {
      local_nn_distance = node_to_visit->search(
          search_cache.query_series_ptr_, search_cache.m256_fetch_cache_.get());
      node_to_visit->neurofilter_->push_example(search_cache.answer_.get_bsf(), local_nn_distance);
    }

    pthread_mutex_lock(search_cache.log_mutex_);
    *search_cache.visited_node_counter_ += 1;
    *search_cache.visited_series_counter_ += node_to_visit->get_size();
    pthread_mutex_unlock(search_cache.log_mutex_);

    if (local_nn_distance < local_bsf) {
      pthread_rwlock_wrlock(search_cache.answer_lock_);
      search_cache.answer_.push_bsf(local_nn_distance);
      pthread_rwlock_unlock(search_cache.answer_lock_);

      local_bsf = search_cache.answer_.get_bsf();

      MALAT_LOG(search_cache.logger_.logger, trivial::info) << boost::format(
            "nf query %d thread %d update bsf %f after node %d series %s")
            % search_cache.query_id_
            % search_cache.thread_id_
            % local_nn_distance
            % *search_cache.visited_node_counter_
            % *search_cache.visited_series_counter_;
    }
  }
}

RESPONSE dstree::Index::nf_collect_mthread() {
  auto m256_fetch_cache = std::unique_ptr<VALUE_TYPE>(static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8)));

  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;

  auto answer = std::make_shared<dstree::Answer>(config_->n_nearest_neighbor_, -1);

  std::unique_ptr<pthread_rwlock_t> answer_lock = std::make_unique<pthread_rwlock_t>();
  std::unique_ptr<pthread_rwlock_t> leaf_pq_lock = std::make_unique<pthread_rwlock_t>();
  std::unique_ptr<pthread_mutex_t> log_mutex = std::make_unique<pthread_mutex_t>();

  std::vector<std::unique_ptr<SearchCache>> search_caches;

  for (ID_TYPE thread_id = 0; thread_id < config_->nf_collect_nthread_; ++thread_id) {
    search_caches.push_back(std::make_unique<SearchCache>(thread_id,
                                                          std::ref(*logger_),
                                                          std::ref(*config_),
                                                          std::ref(*answer),
                                                          answer_lock.get(),
                                                          std::ref(leaf_min_heap_),
                                                          leaf_pq_lock.get(),
                                                          &visited_node_counter,
                                                          &visited_series_counter,
                                                          log_mutex.get()));
  }

  for (ID_TYPE query_id = 0; query_id < config_->nf_train_nexample_; ++query_id) {
    VALUE_TYPE *series_ptr = nf_train_query_ptr_ + config_->series_length_ * query_id;

    visited_node_counter = 0;
    visited_series_counter = 0;
    answer->reset(query_id);

    std::shared_ptr<dstree::Node> resident_node = root_;

    while (!resident_node->is_leaf()) {
      resident_node = resident_node->route(series_ptr);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_->nf_collect_nthread_; ++thread_id) {
      search_caches[thread_id]->query_id_ = query_id;
      search_caches[thread_id]->query_series_ptr_ = series_ptr;
    }

    VALUE_TYPE local_nn_distance = resident_node->search(series_ptr, m256_fetch_cache.get());
    resident_node->neurofilter_->push_example(answer->get_bsf(), local_nn_distance);

    visited_node_counter += 1;
    visited_series_counter += resident_node->get_size();

    if (answer->is_bsf(local_nn_distance)) {
      MALAT_LOG(logger_->logger, trivial::info) << boost::format(
            "nf query %d update bsf %f after node %d series %s")
            % query_id
            % local_nn_distance
            % visited_node_counter
            % visited_series_counter;

      answer->push_bsf(local_nn_distance);
    }

    std::stack<std::shared_ptr<dstree::Node>> node_stack;
    node_stack.push(root_);

    while (!node_stack.empty()) {
      std::shared_ptr<dstree::Node> node_to_visit = node_stack.top();
      node_stack.pop();

      if (node_to_visit->is_leaf()) {
        if (node_to_visit->get_id() != resident_node->get_id()) {
          VALUE_TYPE child_lower_bound_EDsquare = node_to_visit->cal_lower_bound_EDsquare(series_ptr);
          leaf_min_heap_.push(std::make_tuple(node_to_visit, child_lower_bound_EDsquare));
        }
      } else {
        for (const auto &child_node : node_to_visit->children_) {
          node_stack.push(child_node);
        }
      }
    }

    std::vector<std::thread> threads;

    for (ID_TYPE thread_id = 0; thread_id < config_->nf_collect_nthread_; ++thread_id) {
      std::thread current_thread(search_thread_F, std::ref(*search_caches[thread_id]));
      threads.push_back(std::move(current_thread));
    }

    for (ID_TYPE thread_id = 0; thread_id < config_->nf_collect_nthread_; ++thread_id) {
      threads[thread_id].join();
    }

#ifdef DEBUG
//#ifndef DEBUGGED
    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "nf query %d visited %d nodes %d series")
          % query_id
          % visited_node_counter
          % visited_series_counter;
//#endif
#endif
  }

  return SUCCESS;
}

RESPONSE dstree::Index::nf_train() {
  while (!filter_cache_.empty()) {
    std::reference_wrapper<Filter> filter = filter_cache_.top();
    filter_cache_.pop();

    filter.get().train();
  }

  return SUCCESS;
}

struct TrainCache {
  TrainCache(ID_TYPE thread_id,
             upcite::Logger &logger,
             dstree::Config &config,
             at::cuda::CUDAStream stream,
             std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache,
             pthread_rwlock_t *filter_cache_lock) :
      thread_id_(thread_id),
      logger_(logger),
      config_(config),
      stream_(stream),
      filter_cache_(filter_cache),
      filter_cache_lock_(filter_cache_lock) {}
  ~TrainCache() = default;

  ID_TYPE thread_id_;

  upcite::Logger &logger_;
  dstree::Config &config_;

  at::cuda::CUDAStream stream_;

  std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache_;
  pthread_rwlock_t *filter_cache_lock_;
};

void train_thread_F(TrainCache &train_cache) {
  at::cuda::setCurrentCUDAStream(train_cache.stream_);
  at::cuda::CUDAStreamGuard guard(train_cache.stream_); // compiles with cuda

  while (true) {
    pthread_rwlock_wrlock(train_cache.filter_cache_lock_);
    if (train_cache.filter_cache_.empty()) {
      pthread_rwlock_unlock(train_cache.filter_cache_lock_);

      break;
    } else {
      std::reference_wrapper<dstree::Filter> filter = train_cache.filter_cache_.top();
      train_cache.filter_cache_.pop();

      pthread_rwlock_unlock(train_cache.filter_cache_lock_);

      filter.get().train();
    }
  }
}

RESPONSE dstree::Index::nf_train_mthread() {
  // TODO enable multithread train on CPU
  assert(config_->nf_train_is_gpu_);
  assert(torch::cuda::is_available());

  std::stack<std::reference_wrapper<dstree::Filter>> filters;
  std::unique_ptr<pthread_rwlock_t> filter_stack_lock = std::make_unique<pthread_rwlock_t>();

  std::stack<std::reference_wrapper<dstree::Node>> node_stack;
  node_stack.push(std::ref(*root_));

  while (!node_stack.empty()) {
    std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
    node_stack.pop();

    if (node_to_visit.get().is_leaf()) {
      if (node_to_visit.get().neurofilter_ != nullptr) {
        filters.push(std::ref(*node_to_visit.get().neurofilter_));
      }
    } else {
      for (const auto &child_node : node_to_visit.get().children_) {
        node_stack.push(std::ref(*child_node));
      }
    }
  }

  std::vector<std::unique_ptr<TrainCache>> train_caches;

  for (ID_TYPE thread_id = 0; thread_id < config_->nf_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_->nf_device_id_);

    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "thread %d stream id = %d, query = %d, priority = %d")
          % thread_id
          % new_stream.id()
          % new_stream.query()
          % new_stream.priority();

    train_caches.push_back(std::make_unique<TrainCache>(thread_id,
                                                        std::ref(*logger_),
                                                        std::ref(*config_),
                                                        std::move(new_stream),
                                                        std::ref(filters),
                                                        filter_stack_lock.get()));
  }

  std::vector<std::thread> threads;

  for (ID_TYPE thread_id = 0; thread_id < config_->nf_train_nthread_; ++thread_id) {
    std::thread current_thread(train_thread_F, std::ref(*train_caches[thread_id]));
    threads.push_back(std::move(current_thread));
  }

  for (ID_TYPE thread_id = 0; thread_id < config_->nf_train_nthread_; ++thread_id) {
    threads[thread_id].join();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::train() {
  if (!fs::exists(config_->nf_query_filepath_)) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "neurofilter train query filepath %s does not exist")
          % config_->nf_query_filepath_;
  } else {
    std::ifstream query_fin(config_->nf_query_filepath_, std::ios::in | std::ios::binary);
    if (!query_fin.good()) {
      MALAT_LOG(logger_->logger, trivial::error) << boost::format(
            "neurofilter train query filepath %s cannot open")
            % config_->nf_query_filepath_;
    }

    auto query_nbytes = static_cast<ID_TYPE>(
        sizeof(VALUE_TYPE)) * config_->series_length_ * config_->nf_train_nexample_;
    nf_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

    query_fin.read(reinterpret_cast<char *>(nf_train_query_ptr_), query_nbytes);

    if (query_fin.fail()) {
      MALAT_LOG(logger_->logger, trivial::error) << boost::format(
            "cannot read %d bytes from %s")
            % config_->nf_query_filepath_
            % query_nbytes;

      std::free(nf_train_query_ptr_);
      nf_train_query_ptr_ = nullptr;
    }
  }

  if (nf_train_query_ptr_ == nullptr) {
    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "generate synthetic neurofilter train queries");

    // TODO
  }

  if (config_->nf_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_->nf_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  nf_train_query_tsr_ = torch::from_blob(nf_train_query_ptr_,
                                         {config_->nf_train_nexample_, config_->series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  nf_train_query_tsr_ = nf_train_query_tsr_.to(*device_);

  ID_TYPE filter_id = 0;
  nf_initialize(root_, &filter_id);

  MALAT_LOG(logger_->logger, trivial::info) << boost::format(
        "initialized %d neurofilters")
        % filter_id;

  if (config_->nf_train_is_mthread_) {
    nf_collect_mthread();
  } else {
    nf_collect();
  }

  if (config_->nf_train_is_mthread_) {
    nf_train_mthread();
  } else {
    nf_train();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::load() {
  // TODO

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare>(
      Compare(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  return FAILURE;
}

RESPONSE dstree::Index::dump() {
  // TODO
  return FAILURE;
}

RESPONSE dstree::Index::search() {
  if (!fs::exists(config_->query_filepath_)) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "query filepath does not exist = %s")
          % config_->query_filepath_;

    return FAILURE;
  }

  std::ifstream query_fin(config_->query_filepath_, std::ios::in | std::ios::binary);
  if (!query_fin.good()) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "query filepath cannot open = %s")
          % config_->db_filepath_;

    return FAILURE;
  }

  auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_->series_length_ * config_->query_nseries_;
  auto query_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_nbytes));

  query_fin.read(reinterpret_cast<char *>(query_buffer), query_nbytes);

  if (query_fin.fail()) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "cannot read %d bytes from %s")
          % config_->db_filepath_
          % query_nbytes;

    return FAILURE;
  }

  for (ID_TYPE query_id = 0; query_id < config_->query_nseries_; ++query_id) {
    VALUE_TYPE *series_ptr = query_buffer + config_->series_length_ * query_id;
    search(query_id, series_ptr);
  }

  return SUCCESS;
}

RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *series_ptr) {
  auto answer = std::make_shared<dstree::Answer>(config_->n_nearest_neighbor_, query_id);

  if (config_->require_neurofilter_) {
    nf_query_tsr_ = torch::from_blob(series_ptr,
                                     {1, config_->series_length_},
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
    nf_query_tsr_ = nf_query_tsr_.to(*device_);
  }

  std::shared_ptr<dstree::Node> resident_node = root_;

  while (!resident_node->is_leaf()) {
    resident_node = resident_node->route(series_ptr);
  }

  ID_TYPE resident_node_id = resident_node->get_id();

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;

  resident_node->search(series_ptr, answer, visited_node_counter, visited_series_counter);

  if (config_->is_exact_search_) {
    leaf_min_heap_.push(std::make_tuple(root_, 0));

    std::shared_ptr<dstree::Node> node_to_visit;
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()) {
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger_->logger, trivial::debug) << boost::format(
            "query %d node_id %d leaf_min_heap_.size %d node2visit_lbdistance %.3f bsf %.3f")
            % answer->query_id_
            % node_to_visit->get_id()
            % leaf_min_heap_.size()
            % node2visit_lbdistance
            % answer->get_bsf();
#endif
#endif

      leaf_min_heap_.pop();

      if (node_to_visit->is_leaf()) {
        if (visited_node_counter < config_->search_max_nnode_
            && visited_series_counter < config_->search_max_nseries_) {
          if (node_to_visit->get_id() != resident_node_id) {
            if (config_->is_ground_truth_ || answer->is_bsf(node2visit_lbdistance)) {
//              if (node_to_visit->neurofilter_ != nullptr
//                  && node_to_visit->neurofilter_->infer(nf_query_tsr_) < answer->get_bsf()) {
//                node_to_visit->search(series_ptr, answer, visited_node_counter, visited_series_counter);
//              } else {
//                nfpruned_node_counter += 1;
//                nfpruned_series_counter += node_to_visit->get_size();
//              }
              if (node_to_visit->neurofilter_ != nullptr) {
                VALUE_TYPE predicted_nn_distance = node_to_visit->neurofilter_->infer(nf_query_tsr_);

#ifdef DEBUG
//#ifndef DEBUGGED
                MALAT_LOG(logger_->logger, trivial::debug) << boost::format(
                      "query %d node_id %d predicted_nn_distance %.3f bsf %.3f")
                      % answer->query_id_
                      % node_to_visit->get_id()
                      % predicted_nn_distance
                      % answer->get_bsf();
//#endif
#endif

                if (predicted_nn_distance < answer->get_bsf()) {
                  node_to_visit->search(series_ptr, answer, visited_node_counter, visited_series_counter);
                } else {
                  nfpruned_node_counter += 1;
                  nfpruned_series_counter += node_to_visit->get_size();
                }
              } else {
                node_to_visit->search(series_ptr, answer, visited_node_counter, visited_series_counter);
              }
            }
          }
        }
      } else {
        for (const auto &child_node : node_to_visit->children_) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node->cal_lower_bound_EDsquare(series_ptr);

          if (config_->is_ground_truth_ || answer->is_bsf(child_lower_bound_EDsquare)) {
            leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
          }
        }
      }
    }
  }

  MALAT_LOG(logger_->logger, trivial::info) << boost::format(
        "query %d visited %d nodes %d series")
        % query_id
        % visited_node_counter
        % visited_series_counter;

  if (config_->require_neurofilter_) {
    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "query %d neurofilters pruned %d nodes %d series")
          % query_id
          % nfpruned_node_counter
          % nfpruned_series_counter;
  }

  ID_TYPE nnn_to_return = config_->n_nearest_neighbor_;
  while (!answer->empty()) {
    MALAT_LOG(logger_->logger, trivial::info) << boost::format(
          "query %d nn %d = %f")
          % query_id
          % nnn_to_return
          % answer->pop_bsf();

    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0) {
    return FAILURE;
  }

  return SUCCESS;
}
