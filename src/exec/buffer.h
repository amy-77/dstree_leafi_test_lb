//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_BUFFER_H_
#define DSTREE_SRC_EXEC_BUFFER_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <fstream>

#include <spdlog/spdlog.h>

#include "global.h"
#include "config.h"
//#include "logger.h"
#include "eapca.h"

namespace upcite {
namespace dstree {

class Buffer {
 public:
  Buffer(bool is_on_disk,
         ID_TYPE capacity,
         ID_TYPE series_length,
         VALUE_TYPE *global_buffer,
         std::string filepath);
  ~Buffer();

  RESPONSE insert(ID_TYPE offset);
  RESPONSE flush(VALUE_TYPE *load_buffer,
                 VALUE_TYPE *flush_buffer,
                 ID_TYPE series_length);
  RESPONSE clean(bool if_remove_cache = false);

  ID_TYPE get_offset(ID_TYPE node_series_id) const { return offsets_[node_series_id]; }

  const VALUE_TYPE *get_next_series_ptr();
  RESPONSE reset();

  bool is_full() const { return capacity_ > 0 && offsets_.size() == capacity_; }
  ID_TYPE size() const { return size_; }

 private:
  bool is_on_disk_;
  ID_TYPE capacity_, size_;
  ID_TYPE series_length_;

  std::vector<ID_TYPE> offsets_;
  VALUE_TYPE *global_buffer_, *local_buffer_;

  ID_TYPE next_series_id_;

  std::string filepath_;
};

class BufferManager {
 public:
  BufferManager(Config &config);
  ~BufferManager();

  RESPONSE load_batch();

  VALUE_TYPE *get_series_ptr(ID_TYPE series_batch_id) const {
    return batch_load_buffer_ + config_.get().series_length_ * series_batch_id;
  }

  VALUE_TYPE *get_sketch_ptr(ID_TYPE series_batch_id) const {
    return batch_load_sketch_buffer_ + config_.get().sketch_length_ * series_batch_id;
  }

  EAPCA &get_series_eapca(ID_TYPE series_batch_id) const {
    return *batch_eapca_[series_batch_id];
  }

  RESPONSE emplace_series_eapca(std::unique_ptr<EAPCA> eapca) {
    batch_eapca_.emplace_back(std::move(eapca));
    return SUCCESS;
  }

  ID_TYPE load_buffer_size() const { return batch_nseries_; }
  bool is_fully_loaded() const { return loaded_nseries_ == config_.get().db_nseries_; }

  RESPONSE flush();
  RESPONSE clean(bool if_remove_cache = false);

  Buffer &create_node_buffer(ID_TYPE node_id);

 private:
  std::reference_wrapper<Config> config_;

  std::ifstream db_fin_;
  ID_TYPE batch_series_offset_, batch_nseries_, loaded_nseries_;
  VALUE_TYPE *batch_load_buffer_;
  std::vector<std::unique_ptr<EAPCA>> batch_eapca_;

  std::ifstream sketch_fin_;
  VALUE_TYPE *batch_load_sketch_buffer_;

  VALUE_TYPE *batch_flush_buffer_;

  std::vector<std::unique_ptr<Buffer>> node_buffers_;
  std::unordered_map<ID_TYPE, ID_TYPE> node_to_buffer_;
  std::unordered_map<ID_TYPE, ID_TYPE> buffer_to_node_;
};

}
}

#endif //DSTREE_SRC_EXEC_BUFFER_H_
