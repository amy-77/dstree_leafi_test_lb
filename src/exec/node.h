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
#include "filter.h"

namespace upcite {
namespace dstree {

class Node {
 public:
//  Node() = default; // TODO check
  Node(Config &config,
       upcite::Logger &logger,
       dstree::BufferManager &buffer_manager,
       ID_TYPE depth,
       ID_TYPE id);
  Node(Config &config,
       upcite::Logger &logger,
       dstree::BufferManager &buffer_manager,
       ID_TYPE depth,
       ID_TYPE id,
       EAPCAEnvelope &eapca_envelope);
  ~Node() = default;

  Node &route(const VALUE_TYPE *series_ptr) const;
  Node &route(dstree::EAPCA &series_eapca) const;

  RESPONSE enqueue_leaf(std::vector<std::reference_wrapper<Node>> &leaves);
//  RESPONSE enqueue_children(std::vector<std::shared_ptr<Node>> &leaves);

  RESPONSE insert(ID_TYPE series_id,
                  dstree::EAPCA &series_eapca);

  RESPONSE split(dstree::BufferManager &buffer_manager,
                 ID_TYPE first_child_id);

  RESPONSE search(const VALUE_TYPE *query_series_ptr,
                  Answer &answer,
                  ID_TYPE &visited_node_counter,
                  ID_TYPE &visited_series_counter) const;

  VALUE_TYPE search(const VALUE_TYPE *query_series_ptr,
                    VALUE_TYPE *m256_fetch_cache,
                    VALUE_TYPE bsf_distance = -1) const;

  VALUE_TYPE cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr) const {
    return eapca_envelope_->cal_lower_bound_EDsquare(series_ptr, logger_);
  }

  ID_TYPE get_id() const { return id_; }
  ID_TYPE get_size() const { return nseries_; }

  bool is_full() const { return nseries_ == config_.get().leaf_max_nseries_; }
  bool is_leaf() const { return children_.empty(); }

  RESPONSE log();

  // TODO make private
  std::vector<std::unique_ptr<Node>> children_;

  // TODO make private
  std::unique_ptr<Filter> neurofilter_;

 private:
  ID_TYPE depth_, id_;
  ID_TYPE nseries_;

  std::reference_wrapper<Config> config_;
  std::reference_wrapper<upcite::Logger> logger_;

  std::unique_ptr<EAPCAEnvelope> eapca_envelope_;
  std::reference_wrapper<Buffer> buffer_;

  std::unique_ptr<Split> split_;
//  std::shared_ptr<Node> parent_;
//  std::vector<std::shared_ptr<Node>> children_;
};

}
}

#endif //DSTREE_NODE_H
