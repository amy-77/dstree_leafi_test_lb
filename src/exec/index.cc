//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <tuple>
#include <memory>
#include <random>
#include <algorithm>
#include <immintrin.h>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "vec.h"
#include "eapca.h"
#include "answer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Index::Index(Config &config) :
    config_(config),
    nnode_(0),
    nleaf_(0),
    filter_train_query_ptr_(nullptr),
    allocator_(nullptr),
    navigator_(nullptr) {
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_);

  root_ = std::make_unique<dstree::Node>(config_, *buffer_manager_, 0, nnode_);
  nnode_ += 1, nleaf_ += 1;

  if (config_.get().filter_infer_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                              static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  if (config_.get().require_neurofilter_) {
    allocator_ = std::make_unique<dstree::Allocator>(config);
  }
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

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
      CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_) {
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
    target_node = target_node.get().route(series_eapca, true);
  }

  if (target_node.get().is_full()) {
    target_node.get().split(*buffer_manager_, nnode_);
    nnode_ += config_.get().node_nchild_, nleaf_ += config_.get().node_nchild_ - 1;

    target_node = target_node.get().route(series_eapca, true);
  }

  return target_node.get().insert(batch_series_id, series_eapca);
}

RESPONSE dstree::Index::filter_initialize(dstree::Node &node,
                                          ID_TYPE *filter_id) {
  if (node.is_leaf()) {
    node.add_filter(*filter_id, filter_train_query_tsr_);

    filter_cache_.push(node.get_filter());
    *filter_id += 1;
  } else {
    for (auto child_node : node) {
      filter_initialize(child_node, filter_id);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Index::filter_deactivate(dstree::Node &node) {
  if (node.is_leaf()) {
    if (node.has_active_filter()) {
      node.deactivate_filter();
    }
  } else {
    for (auto child_node : node) {
      filter_deactivate(child_node);
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
    auto answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }

    VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, m256_fetch_cache);
    resident_node.get().push_filter_example(answer->get_bsf(), local_nn_distance, 0);

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    if (answer->is_bsf(local_nn_distance)) {
      spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
                   query_id, local_nn_distance, visited_node_counter, visited_series_counter);

      answer->push_bsf(local_nn_distance, resident_node.get().get_id());
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
        if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
          local_nn_distance = node_to_visit.get().search(series_ptr, m256_fetch_cache);
          node_to_visit.get().push_filter_example(answer->get_bsf(), local_nn_distance,
                                                  node2visit_lbdistance);

          visited_node_counter += 1;
          visited_series_counter += node_to_visit.get().get_size();

          if (answer->is_bsf(local_nn_distance)) {
            spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}",
                         query_id, local_nn_distance, visited_node_counter, visited_series_counter);

            answer->push_bsf(local_nn_distance, node_to_visit.get().get_id());
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
              dstree::Answers *answer,
              pthread_mutex_t *answer_mutex,
              std::reference_wrapper<std::priority_queue<dstree::NODE_DISTNCE,
                                                         std::vector<dstree::NODE_DISTNCE>,
                                                         dstree::CompareDecrNodeDist>> leaf_min_heap,
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

  dstree::Answers *answer_;
  pthread_mutex_t *answer_mutex_;

  std::reference_wrapper<std::priority_queue<
      dstree::NODE_DISTNCE, std::vector<dstree::NODE_DISTNCE>, dstree::CompareDecrNodeDist>> leaf_min_heap_;
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

  while (true) {
    pthread_mutex_lock(search_cache.leaf_pq_mutex_);

    if (search_cache.leaf_min_heap_.get().empty()) {
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
      break;
    } else {
      std::tie(node_to_visit, node2visit_lbdistance) = search_cache.leaf_min_heap_.get().top();
      search_cache.leaf_min_heap_.get().pop();

      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
    }

    // for a more precise bsf distance
    pthread_mutex_lock(search_cache.answer_mutex_);
    VALUE_TYPE global_bsf = search_cache.answer_->get_bsf();
    pthread_mutex_unlock(search_cache.answer_mutex_);

    if (node_to_visit.get().has_filter()) {
      VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
          search_cache.query_series_ptr_, *search_cache.answer_, search_cache.answer_mutex_);

      node_to_visit.get().push_filter_example(global_bsf, local_nn_distance, node2visit_lbdistance);

      pthread_mutex_lock(search_cache.log_mutex_);
      *search_cache.visited_node_counter_ += 1;
      *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
      pthread_mutex_unlock(search_cache.log_mutex_);
    } else if (node2visit_lbdistance <= global_bsf) {
      VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
          search_cache.query_series_ptr_, *search_cache.answer_, search_cache.answer_mutex_);

      pthread_mutex_lock(search_cache.log_mutex_);
      *search_cache.visited_node_counter_ += 1;
      *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
      pthread_mutex_unlock(search_cache.log_mutex_);
    }
  }
}

RESPONSE dstree::Index::filter_collect_mthread() {
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(
      sizeof(__m256), 8 * config_.get().filter_collect_nthread_));

  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;

  std::unique_ptr<Answers> answer = nullptr;
  if (config_.get().navigator_is_learned_) {
    // TODO
    assert(!config_.get().require_neurofilter_);
    answer = std::make_unique<dstree::Answers>(config_.get().navigator_train_k_nearest_neighbor_, -1);
  } else {
    answer = std::make_unique<dstree::Answers>(config_.get().n_nearest_neighbor_, -1);
  }

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

    VALUE_TYPE global_bsf_distance = answer->get_bsf();
    VALUE_TYPE local_nn_distance = resident_node.get().search_mt(series_ptr,
                                                                 std::ref(*answer.get()),
                                                                 answer_mutex.get());
    if (config_.get().require_neurofilter_) {
      resident_node.get().push_filter_example(global_bsf_distance, local_nn_distance, 0);
    }

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

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

    train_answers_.emplace_back(dstree::Answers(*answer));

    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    if (config_.get().navigator_is_learned_) {
      nnn_to_return = config_.get().navigator_train_k_nearest_neighbor_;
    }

    while (!answer->empty()) {
      auto answer_i = answer->pop_answer();

      if (answer_i.node_id_ > 0) {
        spdlog::info("query {:d} nn {:d} = {:.3f}, node {:d}",
                     query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
      } else {
        spdlog::info("query {:d} nn {:d} = {:.3f}",
                     query_id, nnn_to_return, answer_i.nn_dist_);
      }

      nnn_to_return -= 1;
    }
  }

  std::free(m256_fetch_cache);
  return SUCCESS;
}

RESPONSE dstree::Index::filter_allocate(bool to_assign, bool reassign) {
  std::stack<std::reference_wrapper<dstree::Node>> node_cache;
  node_cache.push(std::ref(*root_));

  while (!node_cache.empty()) {
    std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
    node_cache.pop();

    if (node_to_visit.get().is_leaf()) {
      FilterInfo filter_info(node_to_visit);
      filter_info.external_pruning_probability_ = node_to_visit.get().get_envelop_pruning_frequency();

      allocator_->push_instance(filter_info);
    } else {
      for (auto child_node : node_to_visit.get()) {
        node_cache.push(child_node);
      }
    }
  }

  if (to_assign) {
    return allocator_->assign();
  } else if (reassign) {
    return allocator_->reassign();
  } else {
    return SUCCESS;
  }
}

RESPONSE dstree::Index::filter_train() {
  while (!filter_cache_.empty()) {
    std::reference_wrapper<Filter> filter = filter_cache_.top();
    filter_cache_.pop();

    if (filter.get().is_active()) {
      filter.get().train();
    }
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

  // TODO remove ref to filter; use node instead
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

      if (filter.get().is_active()) {
        filter.get().train();
      }
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
#ifndef DEBUGGED
  spdlog::debug("indexing filters.size = {:d}", filters.size());
#endif
#endif

  std::vector<std::unique_ptr<TrainCache>> train_caches;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    spdlog::info("train thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
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

RESPONSE dstree::Index::train(bool is_retrain) {
  if (!fs::exists(config_.get().filter_query_filepath_)) {
    assert(!is_retrain); // not applicable to loaded filters

    if (!config_.get().filter_query_filepath_.empty()) {
      spdlog::error("filter train query filepath {:s} does not exist", config_.get().filter_query_filepath_);
      return FAILURE;
    }

    ID_TYPE num_synthetic_queries = root_->get_num_synthetic_queries(allocator_->get_node_size_threshold());

    ID_TYPE query_set_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE))
        * config_.get().series_length_ * num_synthetic_queries;
    filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_set_nbytes));

    ID_TYPE num_generated_queries = 0;
    root_->synthesize_query(filter_train_query_ptr_, num_generated_queries, allocator_->get_node_size_threshold());
    assert(num_generated_queries == num_synthetic_queries);

    config_.get().filter_train_nexample_ = num_synthetic_queries;

    std::string filter_query_filepath = config_.get().index_dump_folderpath_ + config_.get().filter_query_filename_;
    std::ofstream query_fout(filter_query_filepath, std::ios::binary | std::ios_base::app);
    query_fout.write(reinterpret_cast<char *>(filter_train_query_ptr_), query_set_nbytes);
    query_fout.close();

    spdlog::info("filter train generated {:d} filter train queries", config_.get().filter_train_nexample_);
  } else {
    auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE))
        * config_.get().series_length_ * config_.get().filter_train_nexample_;
    filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

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

  // support difference devices for training and inference
  if (config_.get().require_neurofilter_) {
    if (config_.get().filter_train_is_gpu_) {
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  } else if (config_.get().navigator_is_learned_) {
    if (config_.get().navigator_is_gpu_) {
      device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  filter_train_query_tsr_ = torch::from_blob(filter_train_query_ptr_,
                                             {config_.get().filter_train_nexample_, config_.get().series_length_},
                                             torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  filter_train_query_tsr_ = filter_train_query_tsr_.to(*device_);

  if (config_.get().require_neurofilter_) {
    if (is_retrain) {
      filter_deactivate(*root_);
    } else {
      // initialize filters
      ID_TYPE filter_id = 0;
      filter_initialize(*root_, &filter_id);
      spdlog::info("initialized {:d} filters", filter_id);
    }
  }

  if (!is_retrain) {
    // collect filter training data, i.e., the bsf distances, nn distances, low-bound distances
    if (config_.get().filter_train_is_mthread_) {
      filter_collect_mthread();
    } else {
      filter_collect();
    }
  }

  if (config_.get().require_neurofilter_) {
    // allocate filters among nodes (and activate them)
    filter_allocate(true);

    // train all filter model
    if (config_.get().filter_train_is_mthread_) {
      filter_train_mthread();
    } else {
      filter_train();
    }

    // support difference devices for training and inference
    if (config_.get().filter_infer_is_gpu_) {
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  if (config_.get().navigator_is_learned_) {
    leaf_nodes_.reserve(nleaf_);

    auto node_pos_to_id = make_reserved<ID_TYPE>(nleaf_);
    std::unordered_map<ID_TYPE, ID_TYPE> node_id_to_pos;
    node_id_to_pos.reserve(nleaf_ * 2);

    std::stack<std::reference_wrapper<dstree::Node>> node_cache;
    node_cache.push(std::ref(*root_));

    while (!node_cache.empty()) {
      std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
      node_cache.pop();

      if (node_to_visit.get().is_leaf()) {
        leaf_nodes_.push_back(node_to_visit);

        node_pos_to_id.push_back(node_to_visit.get().get_id());
        node_id_to_pos[node_to_visit.get().get_id()] = leaf_nodes_.size() - 1;
      } else {
        for (auto child_node : node_to_visit.get()) {
          node_cache.push(child_node);
        }
      }
    }

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("navigator nleaf_ = {:d}", nleaf_);
#endif
#endif

    auto nn_residence_distributions = make_reserved<VALUE_TYPE>(config_.get().filter_train_nexample_ * nleaf_);
    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i) {
      nn_residence_distributions.push_back(0);
    }

    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i) {
      // use copy constructor to avoid destruct train_answers_
      Answers answers = Answers(train_answers_[query_i]);

      while (!answers.empty()) {
        nn_residence_distributions[nleaf_ * query_i + node_id_to_pos[answers.pop_answer().node_id_]] += 1;
      }
    }

    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i) {
      nn_residence_distributions[cell_i] /= config_.get().navigator_train_k_nearest_neighbor_;
    }

#ifdef DEBUG
    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i) {
      spdlog::debug("navigator train query {:d} target = {:s}",
                    query_i, upcite::array2str(nn_residence_distributions.data() + nleaf_ * query_i, nleaf_));
    }
#endif

    navigator_ = std::make_unique<dstree::Navigator>(config_,
                                                     node_pos_to_id,
                                                     filter_train_query_tsr_,
                                                     nn_residence_distributions,
                                                     *device_);

    navigator_->train();
  }

  return SUCCESS;
}

RESPONSE dstree::Index::load() {
  ID_TYPE ifs_buf_size = sizeof(ID_TYPE) * config_.get().leaf_max_nseries_ * 2; // 2x expanded for safety
  void *ifs_buf = std::malloc(ifs_buf_size);

  nnode_ = 0;
  nleaf_ = 0;

  RESPONSE status = root_->load(ifs_buf, std::ref(*buffer_manager_), nnode_, nleaf_);

  if (status == FAILURE) {
    spdlog::info("failed to load index");
    std::free(ifs_buf);
    return FAILURE;
  }

  std::free(ifs_buf);

  // TODO in-memory only; supports on-disk
  buffer_manager_->load_batch();

  leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
      CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  if (config_.get().require_neurofilter_) {
    if (!config_.get().to_load_filters_) {
      train();
    } else {
      if (config_.get().filter_retrain_) {
        train(true);
      } else if (config_.get().filter_reallocate_multi_) {
        // TODO
        filter_allocate(false, true);
      } else if (config_.get().filter_reallocate_single_) {
        filter_allocate(false, true);
      } else {
        // initialize allocator for setting conformal intervals
        filter_allocate(false);
      }

      // support difference devices for training and inference
      if (config_.get().filter_infer_is_gpu_) {
        // TODO support multiple devices
        device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                  static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
      } else {
        device_ = std::make_unique<torch::Device>(torch::kCPU);
      }
    }
  }

  if (config_.get().navigator_is_learned_) {
    train();
  }

  return SUCCESS;
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

  //
  // get confidence intervals based on the required recall, during search
  if (config_.get().filter_is_conformal_ && config_.get().filter_conformal_adjust_confidence_by_recall_) {
    allocator_.get()->set_confidence_from_recall();
  }

  VALUE_TYPE *series_ptr, *sketch_ptr = nullptr;
  for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
    series_ptr = query_buffer + config_.get().series_length_ * query_id;

    if (config_.get().is_sketch_provided_) {
      sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
    }

    if (config_.get().navigator_is_learned_) {
      search_navigated(query_id, series_ptr, sketch_ptr);
    } else {
      search(query_id, series_ptr, sketch_ptr);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr) {
  VALUE_TYPE *route_ptr = series_ptr;
  if (config_.get().is_sketch_provided_) {
    route_ptr = sketch_ptr;
  }

  auto answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);

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

  resident_node.get().search(series_ptr, *answers, visited_node_counter, visited_series_counter);

  if (config_.get().is_exact_search_) {
    leaf_min_heap_.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));

    // WARN undefined behaviour
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *) nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!leaf_min_heap_.empty()) {
      std::tie(node_to_visit, node2visit_lbdistance) = leaf_min_heap_.top();
      leaf_min_heap_.pop();

      if (node_to_visit.get().is_leaf()) {
        if (visited_node_counter < config_.get().search_max_nnode_ &&
            visited_series_counter < config_.get().search_max_nseries_) {
          if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
            if (config_.get().examine_ground_truth_ || answers->is_bsf(node2visit_lbdistance)) {
              if (node_to_visit.get().has_active_filter()) {
                VALUE_TYPE predicted_nn_distance = node_to_visit.get().filter_infer(filter_query_tsr_);

#ifdef DEBUG
#ifndef DEBUGGED
                spdlog::debug("query {:d} node_i {:d} dist {:.3f} bsf {:.3f} pred {:.3f}",
                              answers.get()->query_id_,
                              node_to_visit.get().get_id(),
                              node2visit_lbdistance,
                              answers->get_bsf(),
                              predicted_nn_distance);
#endif
#endif

                if (predicted_nn_distance > answers->get_bsf()) {
                  nfpruned_node_counter += 1;
                  nfpruned_series_counter += node_to_visit.get().get_size();
                } else {
                  node_to_visit.get().search(series_ptr, *answers, visited_node_counter,
                                             visited_series_counter);
                }
              } else {
                node_to_visit.get().search(series_ptr, *answers, visited_node_counter,
                                           visited_series_counter);
              }
            }
          }
        }
      } else {
        for (auto child_node : node_to_visit.get()) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);

          // TODO fix bug: parent LB dist > child LB dist
          // current workaround: do not early prune
          leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));

//          if (config_.get().examine_ground_truth_ || answers->is_bsf(child_lower_bound_EDsquare)) {
//            leaf_min_heap_.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
//          } else {
//#ifdef DEBUG
////#ifndef DEBUGGED
//            spdlog::debug("query {:d} node_i {:d} ({:b}) dist {:.3f} bsf {:.3f}",
//                          answers.get()->query_id_,
//                          child_node.get().get_id(),
//                          child_node.get().is_leaf(),
//                          child_lower_bound_EDsquare,
//                          answers->get_bsf());
//
//            if (!child_node.get().is_leaf()) {
//              for (auto debug_child_node : child_node.get()) {
//                VALUE_TYPE debug_EDsquare = debug_child_node.get().cal_lower_bound_EDsquare(route_ptr);
//
//                spdlog::debug("query {:d} node_i {:d} ({:d}, {:b}) dist {:.3f} bsf {:.3f}",
//                              answers.get()->query_id_,
//                              debug_child_node.get().get_id(),
//                              child_node.get().get_id(),
//                              debug_child_node.get().is_leaf(),
//                              debug_EDsquare,
//                              answers->get_bsf());
//              }
//            }
////#endif
//#endif
//          }
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

  while (!answers->empty()) {
    auto answer = answers->pop_answer();

    if (answer.node_id_ > 0) {
      spdlog::info("query {:d} nn {:d} = {:.3f}, node {:d}",
                   query_id, nnn_to_return, answer.nn_dist_, answer.node_id_);
    } else {
      spdlog::info("query {:d} nn {:d} = {:.3f}",
                   query_id, nnn_to_return, answer.nn_dist_);
    }

    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0) {
    return FAILURE;
  }

  return SUCCESS;
}

RESPONSE dstree::Index::search_navigated(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr) {
  VALUE_TYPE *route_ptr = series_ptr;
  if (config_.get().is_sketch_provided_) {
    route_ptr = sketch_ptr;
  }

  auto answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);

  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_) {
    filter_query_tsr_ = torch::from_blob(series_ptr,
                                         {1, config_.get().series_length_},
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  }

  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

  while (!resident_node.get().is_leaf()) {
    resident_node = resident_node.get().route(route_ptr);
  }

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;

  resident_node.get().search(series_ptr, *answers, visited_node_counter, visited_series_counter);

  if (config_.get().is_exact_search_) {
    auto node_prob = navigator_->infer(filter_query_tsr_);
    auto node_distances = make_reserved<VALUE_TYPE>(nleaf_);

    if (config_.get().navigator_is_combined_) {
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
        if (leaf_nodes_[leaf_i].get().get_id() == resident_node.get().get_id()) {
          node_distances.push_back(constant::MAX_VALUE);
        } else {
          node_distances.push_back(leaf_nodes_[leaf_i].get().cal_lower_bound_EDsquare(route_ptr));
        }
      }

      VALUE_TYPE min_prob = constant::MAX_VALUE, max_prob = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
        if (node_prob[leaf_i] < min_prob) {
          min_prob = node_prob[leaf_i];
        } else if (node_prob[leaf_i] > max_prob) {
          max_prob = node_prob[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
        node_prob[leaf_i] = (node_prob[leaf_i] - min_prob) / (max_prob - min_prob);
      }

      VALUE_TYPE min_lb_dist = constant::MAX_VALUE, max_lb_dist = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
        if (node_distances[leaf_i] < min_lb_dist) {
          min_lb_dist = node_distances[leaf_i];
        } else if (node_distances[leaf_i] > max_lb_dist && node_distances[leaf_i] < constant::MAX_VALUE / 2) {
          max_lb_dist = node_distances[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
        node_prob[leaf_i] = config_.get().navigator_combined_lambda_ * node_prob[leaf_i]
            + (1 - config_.get().navigator_combined_lambda_)
                * (1 - (node_distances[leaf_i] - min_lb_dist) / (max_lb_dist - min_lb_dist));
      }
    }

    auto node_pos_probs = make_reserved<std::tuple<ID_TYPE, VALUE_TYPE>>(nleaf_);

    for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i) {
      if (leaf_nodes_[leaf_i].get().get_id() != resident_node.get().get_id()) {
        node_pos_probs.push_back(std::tuple<ID_TYPE, VALUE_TYPE>(leaf_i, node_prob[leaf_i]));
      }
    }

#ifdef DEBUG
//#ifndef DEBUGGED
    spdlog::debug("query {:d} node_distances = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_distances.data(), nleaf_));

    spdlog::debug("query {:d} node_prob = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_prob.data(), nleaf_));
//#endif
#endif

    std::sort(node_pos_probs.begin(), node_pos_probs.end(), dstree::compDecrProb);

    for (ID_TYPE prob_i = 0; prob_i < node_pos_probs.size(); ++prob_i) {
      ID_TYPE leaf_i = std::get<0>(node_pos_probs[prob_i]);
      auto node_to_visit = leaf_nodes_[leaf_i];

//#ifdef DEBUG
////#ifndef DEBUGGED
//      spdlog::debug("query {:d} leaf_i {:d} ({:d}) dist {:.3f} prob {:.3f} ({:.3f}) bsf {:.3f}",
//                    answers.get()->query_id_,
//                    leaf_i, navigator_->get_id_from_pos(leaf_i),
//                    node_distances[leaf_i],
//                    std::get<1>(node_pos_probs[prob_i]), node_prob[leaf_i],
//                    answers->get_bsf());
////#endif
//#endif

      if (visited_node_counter < config_.get().search_max_nnode_ &&
          visited_series_counter < config_.get().search_max_nseries_) {
        if (config_.get().examine_ground_truth_ || answers->is_bsf(node_distances[leaf_i])) {
          node_to_visit.get().search(series_ptr, *answers, visited_node_counter, visited_series_counter);
        }
      }
    }
  }

  spdlog::info("query {:d} visited {:d} nodes {:d} series",
               query_id, visited_node_counter, visited_series_counter);

  ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

  while (!answers->empty()) {
    auto answer = answers->pop_answer();

    if (answer.node_id_ > 0) {
      spdlog::info("query {:d} nn {:d} = {:.3f}, node {:d}",
                   query_id, nnn_to_return, answer.nn_dist_, answer.node_id_);
    } else {
      spdlog::info("query {:d} nn {:d} = {:.3f}",
                   query_id, nnn_to_return, answer.nn_dist_);
    }

    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0) {
    return FAILURE;
  }

  return SUCCESS;
}
