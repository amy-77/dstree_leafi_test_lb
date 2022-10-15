//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <memory>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include "eapca.h"

namespace dstree = upcite::dstree;

dstree::Index::Index(std::shared_ptr<Config> config, std::shared_ptr<upcite::Logger> logger) :
    config_(std::move(config)),
    logger_(std::move(logger)),
    node_id_accumulator_(0) {
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_, logger_);

  root_ = std::make_shared<dstree::Node>(config_, buffer_manager_, 0, node_id_accumulator_);
  node_id_accumulator_ += 1;
}

RESPONSE dstree::Index::build() {
  while (buffer_manager_->load_batch() == SUCCESS) {
    for (ID_TYPE series_id = 0; series_id < buffer_manager_->load_buffer_size(); ++series_id) {
      insert(series_id);
    }

    buffer_manager_->flush();
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
    target_node = root_->route(series_eapca);
  }

  if (target_node->is_full()) {
    target_node->split(config_, buffer_manager_, node_id_accumulator_, logger_);
    node_id_accumulator_ += config_->node_nchild_;

    target_node = target_node->route(series_eapca);
  }

  return target_node->insert(batch_series_id, series_eapca);
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
  return FAILURE;
}
