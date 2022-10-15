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

namespace upcite {
namespace dstree {

class Node {
 public:
  Node(std::shared_ptr<Config> config,
       const std::unique_ptr<dstree::BufferManager> &buffer_manager,
       ID_TYPE depth,
       ID_TYPE id,
       std::shared_ptr<Node> parent = nullptr,
       const std::shared_ptr<EAPCA_Envelope> &eapca_envelope = nullptr);
  ~Node() = default;

  std::shared_ptr<Node> route(const VALUE_TYPE *series_ptr) const;
  std::shared_ptr<Node> route(const std::shared_ptr<dstree::EAPCA>& series_eapca) const;

  RESPONSE insert(ID_TYPE series_id, const std::shared_ptr<dstree::EAPCA> &series_eapca);

  RESPONSE split(const std::shared_ptr<dstree::Config> &config,
                 const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                 ID_TYPE first_child_id,
                 const std::shared_ptr<upcite::Logger> &logger);

  bool is_full() const { return nseries_ == config_->leaf_max_nseries_; }
  bool is_leaf() const { return children_.empty(); }

  RESPONSE log(const std::shared_ptr<upcite::Logger> &logger);

  std::shared_ptr<Split> split_;

 private:
  ID_TYPE depth_, id_;
  ID_TYPE nseries_;

  std::shared_ptr<Config> config_;
  std::shared_ptr<Buffer> buffer_;

  std::shared_ptr<Node> parent_;
  std::vector<std::shared_ptr<Node>> children_;

  std::shared_ptr<EAPCA_Envelope> eapca_envelope_;
};

}
}

#endif //DSTREE_NODE_H
