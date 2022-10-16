//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "buffer.h"

#include <iostream>
#include <utility>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

dstree::Buffer::Buffer(ID_TYPE capacity, std::string filepath) :
    capacity_(capacity),
    filepath_(std::move(filepath)) {
  offsets_.reserve(capacity_);
}

RESPONSE dstree::Buffer::insert(ID_TYPE offset,
                                const std::shared_ptr<upcite::Logger> &logger) {
  // TODO explicitly managing ID_TYPE * resulted in unexpected change of values in the middle
  offsets_.push_back(offset);

#ifdef DEBUG
#ifndef DEBUGGED
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("%d / %d = %d == %d")
          % size()
          % capacity_
          % offset
          % offsets_[size() - 1];
  }
#endif
#endif

  if (size() > capacity_) {
    std::cout << boost::format("%s: nseries > capacity") % fs::path(filepath_).filename().string() << std::endl; // TODO
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::flush(VALUE_TYPE *load_buffer, VALUE_TYPE *flush_buffer, ID_TYPE series_length) {
  if (size() > 0) {
    auto series_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length;

    for (ID_TYPE i = 0; i < size(); ++i) {
      std::memcpy(flush_buffer + series_length * i, load_buffer + series_length * offsets_[i], series_nbytes);
    }

    std::ofstream fout(filepath_, std::ios::binary | std::ios_base::app);
    fout.write(reinterpret_cast<char *>(flush_buffer), series_nbytes * size());
    fout.close();

    clean();
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::clean(bool if_remove_cache) {
  offsets_.clear();

  if (if_remove_cache) {
    offsets_.shrink_to_fit();
    capacity_ = 0;
  }

  return SUCCESS;
}

dstree::BufferManager::BufferManager(std::shared_ptr<dstree::Config> config, std::shared_ptr<upcite::Logger> logger) :
    config_(std::move(config)),
    logger_(std::move(logger)),
    batch_series_offset_(0),
    loaded_nseries_(0),
    batch_flush_buffer_(nullptr) {
  batch_nseries_ = config_->batch_load_nseries_;
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_->series_length_ * batch_nseries_;
  batch_load_buffer_ = static_cast<VALUE_TYPE *>(std::malloc(batch_nbytes));

  node_buffers_.reserve(config_->default_nbuffer_);

  node_to_buffer_.reserve(config_->default_nbuffer_);
  buffer_to_node_.reserve(config_->default_nbuffer_);
}

dstree::BufferManager::~BufferManager() {
  if (batch_load_buffer_) {
    std::free(batch_load_buffer_);
    batch_load_buffer_ = nullptr;
  }

  if (batch_flush_buffer_) {
    std::free(batch_flush_buffer_);
    batch_flush_buffer_ = nullptr;
  }

  if (db_fin_.is_open()) {
    db_fin_.close();
  }
}

std::shared_ptr<dstree::Buffer> dstree::BufferManager::create_node_buffer(ID_TYPE node_id) {
  auto buffer_id = static_cast<ID_TYPE>(node_buffers_.size());
  std::string buffer_filepath = config_->index_persist_folderpath_ + std::to_string(node_id) +
      config_->index_persist_file_postfix_;

  node_buffers_.push_back(std::make_shared<dstree::Buffer>(config_->leaf_max_nseries_, buffer_filepath));

  node_to_buffer_[node_id] = buffer_id;
  buffer_to_node_[buffer_id] = node_id;

  return node_buffers_[buffer_id];
}

RESPONSE dstree::BufferManager::load_batch() {
  if (loaded_nseries_ == 0) {
    if (!fs::exists(config_->db_filepath_)) {
      MALAT_LOG(logger_->logger, trivial::error)
        << boost::format("database filepath does not exist = %s") % config_->db_filepath_;

      return FAILURE;
    }

    db_fin_.open(config_->db_filepath_, std::ios::in | std::ios::binary);
    if (!db_fin_.good()) {
      MALAT_LOG(logger_->logger, trivial::error)
        << boost::format("database filepath cannot open = %s") % config_->db_filepath_;

      return FAILURE;
    }
  } else if (loaded_nseries_ >= config_->db_nseries_) {
    return FAILURE;
  }

  if (loaded_nseries_ + config_->batch_load_nseries_ > config_->db_nseries_) {
    batch_nseries_ = config_->db_nseries_ - loaded_nseries_;
  }
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_->series_length_ * batch_nseries_;

  batch_series_offset_ = loaded_nseries_;
  auto batch_bytes_offset = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_->series_length_ * batch_series_offset_;

  db_fin_.seekg(batch_bytes_offset);
  db_fin_.read(reinterpret_cast<char *>(batch_load_buffer_), batch_nbytes);

  if (db_fin_.fail()) {
    MALAT_LOG(logger_->logger, trivial::error) << boost::format(
          "cannot read %d bytes from %s at %d")
          % config_->db_filepath_
          % batch_nbytes
          % batch_bytes_offset;

    return FAILURE;
  }

  loaded_nseries_ += batch_nseries_;

  return SUCCESS;
}

RESPONSE dstree::BufferManager::flush() {
  if (batch_flush_buffer_ == nullptr) {
    auto batch_flush_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) *
        config_->series_length_ * config_->leaf_max_nseries_;
    batch_flush_buffer_ = static_cast<VALUE_TYPE *>(std::malloc(batch_flush_nbytes));
  }

  for (const auto &buffer : node_buffers_) {
    buffer->flush(batch_load_buffer_, batch_flush_buffer_, config_->series_length_);
  }

  return SUCCESS;
}

RESPONSE dstree::BufferManager::clean(bool if_remove_cache) {
  for (const auto &buffer : node_buffers_) {
    buffer->clean(if_remove_cache);
  }

  return SUCCESS;
}