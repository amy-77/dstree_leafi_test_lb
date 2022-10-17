//
// Created by Qitong Wang on 2022/10/17.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_TEST_TEST_SHARED_FROM_THIS_H_
#define DSTREE_TEST_TEST_SHARED_FROM_THIS_H_

#include <memory>

#include "buffer.h"
#include "global.h"
#include "logger.h"
#include "config.h"
#include "eapca.h"
#include "split.h"

namespace upcite {
namespace dstree {
namespace test {

class TestNode : public std::enable_shared_from_this<TestNode> {
 public:
  TestNode() { id_ = -1, depth_ = -1, nseries_ = 0; };
  TestNode(const std::shared_ptr<Config> &config,
           const std::unique_ptr<dstree::BufferManager> &buffer_manager,
           ID_TYPE depth,
           ID_TYPE id,
           std::shared_ptr<TestNode> parent = nullptr,
           const std::shared_ptr<EAPCAEnvelope> &eapca_envelope = nullptr);
  ~TestNode() = default;

  std::shared_ptr<TestNode> route(const std::shared_ptr<dstree::EAPCA> &series_eapca,
                                  const std::shared_ptr<upcite::Logger> &logger = nullptr) const;

  RESPONSE insert(ID_TYPE series_id,
                  const std::shared_ptr<dstree::EAPCA> &series_eapca,
                  const std::shared_ptr<upcite::Logger> &logger = nullptr);

  RESPONSE split(const std::shared_ptr<dstree::Config> &config,
                 const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                 ID_TYPE first_child_id,
                 const std::shared_ptr<upcite::Logger> &logger = nullptr);

//  bool is_full() const { return nseries_ == config_->leaf_max_nseries_; }
  bool is_full() const { return nseries_ == 1000; }
//  bool is_leaf() const { return children_.empty(); }
  bool is_leaf() const { return is_leaf_; }

//  std::shared_ptr<Split> split_;

  ID_TYPE depth_, id_;
//  std::shared_ptr<EAPCAEnvelope> eapca_envelope_;

// private:
//  ID_TYPE depth_, id_;
  ID_TYPE nseries_;

  bool is_leaf_;

//  const std::shared_ptr<Config> &config_;
//  std::shared_ptr<Buffer> buffer_;

//  std::shared_ptr<TestNode> parent_;
//  std::vector<std::shared_ptr<TestNode>> children_;

//  std::shared_ptr<EAPCA_Envelope> eapca_envelope_;
};

}
}
}

#endif //DSTREE_TEST_TEST_SHARED_FROM_THIS_H_
