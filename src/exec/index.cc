//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <memory>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "eapca.h"
#include "answer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

dstree::Index::Index(std::shared_ptr<Config> config, std::shared_ptr<upcite::Logger> logger) :
    config_(std::move(config)),
    logger_(std::move(logger)),
    nleaf_(0) {
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_, logger_);

  root_ = std::make_shared<dstree::Node>(config_, buffer_manager_, 0, nleaf_);
  nleaf_ += 1;
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
    target_node = target_node->route(series_eapca, logger_);
  }

  if (target_node->is_full()) {
    target_node->split(config_, buffer_manager_, nleaf_, logger_);
    nleaf_ += config_->node_nchild_;

    target_node = target_node->route(series_eapca, logger_);
  }

  return target_node->insert(batch_series_id, series_eapca, logger_);
}

RESPONSE dstree::Index::load() {
  // TODO
  return FAILURE;
}

RESPONSE dstree::Index::dump() {
  // TODO
  return FAILURE;
}

RESPONSE dstree::Index::search() {
  if (!fs::exists(config_->query_filepath_)) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "database filepath does not exist = %s")
          % config_->query_filepath_;

    return FAILURE;
  }

  std::ifstream query_fin(config_->db_filepath_, std::ios::in | std::ios::binary);
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

  for (ID_TYPE query_id = 0; query_id < config_->series_length_; ++query_id) {
    const VALUE_TYPE *series_ptr = query_buffer + config_->series_length_ * query_id;
    search(query_id, series_ptr);
  }

  return SUCCESS;
}

RESPONSE dstree::Index::search(ID_TYPE query_id, const VALUE_TYPE *series_ptr) {
  if (config_->is_exact_search_) {
    if (leaves_.empty()) {
      leaves_.reserve(nleaf_);
      // TODO

#ifdef DEBUG
      assert(nleaf_ == leaves_.size());
#endif
    }
  }

  auto answer = std::make_shared<dstree::Answer>(query_id);
  std::shared_ptr<Node> target_node = root_;

  while (!target_node->is_leaf()) {
    target_node = target_node->route(series_ptr);
  }

  if (target_node->search(series_ptr, answer)) {
    ID_TYPE resident_node_id = target_node->get_id();
    return FAILURE;
  }

  return SUCCESS;
}
