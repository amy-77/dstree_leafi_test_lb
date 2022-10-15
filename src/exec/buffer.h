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

#include "global.h"
#include "config.h"
#include "logger.h"
#include "eapca.h"

namespace upcite {
namespace dstree {

class Buffer {
 public:
  Buffer(ID_TYPE capacity,
         std::string filepath);
  ~Buffer();

  RESPONSE insert(ID_TYPE offset);
  RESPONSE flush(VALUE_TYPE *load_buffer,
                 VALUE_TYPE *flush_buffer,
                 ID_TYPE series_length);
  RESPONSE clean(bool if_remove_cache = false);

  bool is_full() const { return nseries_ == capacity_; }
  ID_TYPE size() const { return nseries_; }

  ID_TYPE capacity_, nseries_;
  ID_TYPE *offsets_;

 private:
  std::string filepath_;
};

class BufferManager {
 public:
  BufferManager(std::shared_ptr<Config> config,
                std::shared_ptr<upcite::Logger> logger);
  ~BufferManager();

  RESPONSE load_batch();

  VALUE_TYPE *get_series_ptr(ID_TYPE series_batch_id) {
    return batch_load_buffer_ + config_->series_length_ * series_batch_id;
  }

  ID_TYPE load_buffer_size() const { return batch_nseries_; }
  bool is_fully_loaded() const { return loaded_nseries_ == config_->db_nseries_; }

  RESPONSE flush();
  RESPONSE clean(bool if_remove_cache = false);

  std::shared_ptr<Buffer> create_node_buffer(ID_TYPE node_id);

  VALUE_TYPE *batch_load_buffer_;
  std::vector<std::shared_ptr<EAPCA>> batch_eapca_;

 private:
  std::shared_ptr<Config> config_;
  std::shared_ptr<upcite::Logger> logger_;

  ID_TYPE batch_series_offset_, batch_nseries_, loaded_nseries_;
  std::ifstream db_fin_;
  VALUE_TYPE *batch_flush_buffer_;

  std::vector<std::shared_ptr<Buffer>> node_buffers_;
  std::unordered_map<ID_TYPE, ID_TYPE> node_to_buffer_;
  std::unordered_map<ID_TYPE, ID_TYPE> buffer_to_node_;
};

}
}

#endif //DSTREE_SRC_EXEC_BUFFER_H_
