//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "buffer.h"

#include <immintrin.h>
#include <iostream>
#include <utility>

//#include <boost/format.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;

dstree::Buffer::Buffer(bool is_on_disk,
                       ID_TYPE capacity,
                       ID_TYPE series_length,
                       VALUE_TYPE *global_buffer,
                       std::string dump_filepath,
                       std::string load_filepath) :
    is_on_disk_(is_on_disk),
    capacity_(capacity),
    series_length_(series_length),
    global_buffer_(global_buffer),
    local_buffer_(nullptr),
    dump_filepath_(std::move(dump_filepath)),
    load_filepath_(std::move(load_filepath)),
    size_(0),
    next_series_id_(0) {
  offsets_.reserve(16);
}

dstree::Buffer::~Buffer() {
  if (local_buffer_ != nullptr) {
    std::free(local_buffer_);
    local_buffer_ = nullptr;
  }
}

const VALUE_TYPE *dstree::Buffer::get_next_series_ptr() {
  if (is_on_disk_) {
    if (local_buffer_ == nullptr) {
      ID_TYPE local_buffer_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length_ * size_;
      local_buffer_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), local_buffer_nbytes));

      if (!fs::exists(load_filepath_)) {
        spdlog::error("node file {:s} does not exist", load_filepath_);

        return nullptr;
      }

      std::ifstream fin(load_filepath_, std::ios::in | std::ios::binary);
      if (!fin.good()) {
        spdlog::error("node file {:s} cannot open", load_filepath_);

        return nullptr;
      }

      fin.read(reinterpret_cast<char *>(local_buffer_), local_buffer_nbytes);

      if (fin.fail()) {
        spdlog::error("node buffer cannot read {:d} bytes from {:s}", local_buffer_nbytes, load_filepath_);

        return nullptr;
      }

      fin.close();
    }

    if (next_series_id_ < size_) {
      return local_buffer_ + series_length_ * next_series_id_++;
    }
  } else {
    if (next_series_id_ < size_) {
      return global_buffer_ + series_length_ * offsets_[next_series_id_++];
    }
  }

  return nullptr;
}

RESPONSE dstree::Buffer::reset() {
  if (local_buffer_ != nullptr) {
    std::free(local_buffer_);
    local_buffer_ = nullptr;
  }

  next_series_id_ = 0;

  return SUCCESS;
}

RESPONSE dstree::Buffer::insert(ID_TYPE offset) {
  // TODO explicitly managing ID_TYPE * resulted in unexpected change of values in the middle
  offsets_.push_back(offset);
  size_ += 1;

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
    // TODO
    spdlog::error("{:s}: nseries > capacity", fs::path(dump_filepath_).filename().string());

    return FAILURE;
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::flush(VALUE_TYPE *load_buffer, VALUE_TYPE *flush_buffer, ID_TYPE series_length) {
  if (size() > 0) {
    auto series_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * series_length;

    for (ID_TYPE i = 0; i < size(); ++i) {
      std::memcpy(flush_buffer + series_length * i, load_buffer + series_length * offsets_[i], series_nbytes);

    }

    std::ofstream fout(dump_filepath_, std::ios::binary | std::ios_base::app);
    fout.write(reinterpret_cast<char *>(flush_buffer), series_nbytes * size());
    fout.close();

    clean();
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::clean(bool if_remove_cache) {
  offsets_.clear();
  size_ = 0;

  if (if_remove_cache) {
//    offsets_.shrink_to_fit();
    std::vector<ID_TYPE>().swap(offsets_);
    capacity_ = 0;

    if (local_buffer_ != nullptr) {
      free(local_buffer_);
      local_buffer_ = nullptr;
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Buffer::dump() const {
  // TODO support dump in batches

  std::ofstream buffer_ofs(dump_filepath_, std::ios::out | std::ios::binary);
//  assert(buffer_fos.is_open());

  buffer_ofs.write(reinterpret_cast<const char *>(&size_), sizeof(ID_TYPE));
  buffer_ofs.write(reinterpret_cast<const char *>(offsets_.data()), sizeof(ID_TYPE) * offsets_.size());

//  assert(buffer_fos.good());
  buffer_ofs.close();

  return SUCCESS;
}

RESPONSE dstree::Buffer::load(void *ifs_buf) {
  // TODO support load in batches

  std::ifstream buffer_ifs(load_filepath_, std::ios::in | std::ios::binary);
//  assert(buffer_ifs.is_open());

  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  buffer_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_ = ifs_id_buf[0];

  read_nbytes = sizeof(ID_TYPE) * size_;
  buffer_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  offsets_.insert(offsets_.begin(), ifs_id_buf, ifs_id_buf + size_);

//  assert(buffer_ifs.good());
  buffer_ifs.close();

  return SUCCESS;
}

dstree::BufferManager::BufferManager(dstree::Config &config) :
    config_(config),
    batch_series_offset_(0),
    loaded_nseries_(0),
    batch_flush_buffer_(nullptr) {
  batch_nseries_ = config.batch_load_nseries_;
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config.series_length_ * batch_nseries_;
  batch_load_buffer_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

  node_buffers_.reserve(config.default_nbuffer_);

  node_to_buffer_.reserve(config.default_nbuffer_);
  buffer_to_node_.reserve(config.default_nbuffer_);

  if (config_.get().is_sketch_provided_) {
    auto batch_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config.sketch_length_ * batch_nseries_;
    batch_load_sketch_buffer_ = static_cast<VALUE_TYPE *>(malloc(batch_sketch_nbytes));
  }
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

  if (config_.get().is_sketch_provided_) {
    if (batch_load_sketch_buffer_) {
      std::free(batch_load_sketch_buffer_);
      batch_load_sketch_buffer_ = nullptr;
    }

    if (sketch_fin_.is_open()) {
      sketch_fin_.close();
    }
  }
}

dstree::Buffer &dstree::BufferManager::create_node_buffer(ID_TYPE node_id) {
  auto buffer_id = static_cast<ID_TYPE>(node_buffers_.size());
  std::string buffer_filepath = config_.get().dump_data_folderpath_ + std::to_string(node_id) + config_.get().index_dump_file_postfix_;

  std::string load_filepath = buffer_filepath;
  if (config_.get().to_load_index_) {
    load_filepath = config_.get().load_data_folderpath_ + std::to_string(node_id) + config_.get().index_dump_file_postfix_;
  }

  node_buffers_.emplace_back(std::make_unique<dstree::Buffer>(
      config_.get().on_disk_,
      config_.get().leaf_max_nseries_,
      config_.get().series_length_,
      batch_load_buffer_,
      buffer_filepath,
      load_filepath));

  node_to_buffer_[node_id] = buffer_id;
  buffer_to_node_[buffer_id] = node_id;

  return *node_buffers_[buffer_id];
}

RESPONSE dstree::BufferManager::load_batch() {
  if (loaded_nseries_ == 0) {
    if (!fs::exists(config_.get().db_filepath_)) {
      spdlog::error("database filepath does not exist = {:s}", config_.get().db_filepath_);

      return FAILURE;
    }

    db_fin_.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    if (!db_fin_.good()) {
      spdlog::error("database filepath cannot open = {:s}", config_.get().db_filepath_);

      return FAILURE;
    }

    if (config_.get().is_sketch_provided_) {
      if (!fs::exists(config_.get().train_sketch_filepath_)) {
        spdlog::error("database sketch filepath does not exist = {:s}", config_.get().train_sketch_filepath_);

        return FAILURE;
      }

      sketch_fin_.open(config_.get().train_sketch_filepath_, std::ios::in | std::ios::binary);
      if (!sketch_fin_.good()) {
        spdlog::error("database sketch filepath cannot open = {:s}", config_.get().train_sketch_filepath_);

        return FAILURE;
      }
    }
  } else if (loaded_nseries_ >= config_.get().db_nseries_) {
    return FAILURE;
  }

  if (loaded_nseries_ + config_.get().batch_load_nseries_ > config_.get().db_nseries_) {
    batch_nseries_ = config_.get().db_nseries_ - loaded_nseries_;
  }
  auto batch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * batch_nseries_;

  batch_series_offset_ = loaded_nseries_;
  auto batch_bytes_offset =
      static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * batch_series_offset_;

  db_fin_.seekg(batch_bytes_offset);
  db_fin_.read(reinterpret_cast<char *>(batch_load_buffer_), batch_nbytes);

  if (db_fin_.fail()) {
    spdlog::error("cannot read {:d} bytes from {:s} at {:d}",
                  batch_nbytes, config_.get().db_filepath_, batch_bytes_offset);

    return FAILURE;
  }

  if (config_.get().is_sketch_provided_) {
    auto batch_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * batch_nseries_;
    auto batch_sketch_bytes_offset =
        static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * batch_series_offset_;

    sketch_fin_.seekg(batch_sketch_bytes_offset);
    sketch_fin_.read(reinterpret_cast<char *>(batch_load_sketch_buffer_), batch_sketch_nbytes);

    if (sketch_fin_.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s} at {:d}",
                    batch_sketch_nbytes, config_.get().train_sketch_filepath_, batch_sketch_bytes_offset);

      return FAILURE;
    }
  }

  loaded_nseries_ += batch_nseries_;

  return SUCCESS;
}

RESPONSE dstree::BufferManager::flush() {
  if (batch_flush_buffer_ == nullptr) {
    auto batch_flush_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) *
        config_.get().series_length_ * config_.get().leaf_max_nseries_;
    batch_flush_buffer_ = static_cast<VALUE_TYPE *>(std::malloc(batch_flush_nbytes));
  }

  for (const auto &buffer : node_buffers_) {
    buffer->flush(batch_load_buffer_, batch_flush_buffer_, config_.get().series_length_);
  }

  // TODO flush sketches

  return SUCCESS;
}

RESPONSE dstree::BufferManager::clean(bool if_remove_cache) {
  for (const auto &buffer : node_buffers_) {
    buffer->clean(if_remove_cache);
  }

  return SUCCESS;
}
