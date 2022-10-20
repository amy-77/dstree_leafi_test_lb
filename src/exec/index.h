//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_INDEX_H
#define DSTREE_INDEX_H

#include <memory>
#include <vector>

#include "global.h"
#include "config.h"
#include "buffer.h"
#include "node.h"

namespace upcite {
namespace dstree {

using NODE_DISTNCE = std::tuple<std::shared_ptr<dstree::Node>, VALUE_TYPE>;

class Compare {
 public:
  bool operator()(const NODE_DISTNCE &a, const NODE_DISTNCE &b) {
    return std::get<1>(a) > std::get<1>(b);
  }
};

class Index {
 public:
  Index(std::shared_ptr<Config> config, std::shared_ptr<upcite::Logger> logger);
  ~Index() = default;

  RESPONSE build();

  RESPONSE load();
  RESPONSE dump();

  RESPONSE search();
  RESPONSE search(ID_TYPE query_id, const VALUE_TYPE *series_ptr);

 private:
  RESPONSE insert(ID_TYPE batch_series_id);

  std::shared_ptr<Config> config_;
  std::shared_ptr<upcite::Logger> logger_;

  std::unique_ptr<BufferManager> buffer_manager_;

  std::shared_ptr<Node> root_;
  ID_TYPE nnode_, nleaf_;
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare> leaf_min_heap_;
};

}
}

#endif //DSTREE_INDEX_H
