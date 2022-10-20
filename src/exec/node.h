//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_NODE_H
#define DSTREE_NODE_H

#include <memory>

#include "buffer.h"
#include "global.h"
#include "logger.h"
#include "config.h"
#include "eapca.h"
#include "split.h"
#include "answer.h"

namespace upcite {
namespace dstree {

class Node : std::enable_shared_from_this<Node> {
 public:
  Node(std::shared_ptr<Config> config,
       std::shared_ptr<upcite::Logger> logger,
       const std::unique_ptr<dstree::BufferManager> &buffer_manager,
       ID_TYPE depth,
       ID_TYPE id,
       const std::shared_ptr<EAPCAEnvelope> &eapca_envelope = nullptr);
  ~Node() = default;

  std::shared_ptr<Node> route(const VALUE_TYPE *series_ptr) const;
  std::shared_ptr<Node> route(const std::shared_ptr<dstree::EAPCA> &series_eapca) const;

  RESPONSE enqueue_leaf(std::vector<std::shared_ptr<Node>> &leaves);
//  RESPONSE enqueue_children(std::vector<std::shared_ptr<Node>> &leaves);

  RESPONSE insert(ID_TYPE series_id,
                  const std::shared_ptr<dstree::EAPCA> &series_eapca);

  RESPONSE split(const std::shared_ptr<dstree::Config> &config,
                 const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                 ID_TYPE first_child_id);

  RESPONSE search(const VALUE_TYPE *query_series_ptr,
                  std::shared_ptr<Answer> &answer,
                  ID_TYPE &visited_node_counter,
                  ID_TYPE &visited_series_counter) const;

  VALUE_TYPE cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr) const {
    return eapca_envelope_->cal_lower_bound_EDsquare(series_ptr, logger_);
  }

  ID_TYPE get_id() const { return id_; }
  ID_TYPE get_size() const { return nseries_; }

  bool is_full() const { return nseries_ == config_->leaf_max_nseries_; }
  bool is_leaf() const { return children_.empty(); }

  RESPONSE log();

  // TODO make private
  std::vector<std::shared_ptr<Node>> children_;

 private:
  ID_TYPE depth_, id_;
  ID_TYPE nseries_;

  std::shared_ptr<Config> config_;
  std::shared_ptr<upcite::Logger> logger_;

  std::shared_ptr<EAPCAEnvelope> eapca_envelope_;
  std::shared_ptr<Buffer> buffer_;

//  std::shared_ptr<Node> parent_;
  std::shared_ptr<Split> split_;
//  std::vector<std::shared_ptr<Node>> children_;
};

}
}

#endif //DSTREE_NODE_H
