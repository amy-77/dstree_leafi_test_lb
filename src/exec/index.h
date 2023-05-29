//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_INDEX_H
#define DSTREE_INDEX_H

#include <memory>
#include <vector>
#include <stack>

#include <torch/torch.h>

#include "global.h"
#include "config.h"
#include "buffer.h"
#include "node.h"
#include "filter.h"
#include "allocator.h"

namespace upcite {
namespace dstree {

using NODE_DISTNCE = std::tuple<std::reference_wrapper<dstree::Node>, VALUE_TYPE>;

class Compare {
 public:
  bool operator()(const NODE_DISTNCE &a, const NODE_DISTNCE &b) {
    return std::get<1>(a) > std::get<1>(b);
  }
};

class Index {
 public:
  explicit Index(Config &config);
  ~Index();

  RESPONSE build();

  RESPONSE dump() const;
  RESPONSE load();

  RESPONSE search();
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr = nullptr);

 private:
  RESPONSE insert(ID_TYPE batch_series_id);

  RESPONSE train();

  // initialize filter's member variables except the model
  RESPONSE filter_initialize(dstree::Node &node,
                             ID_TYPE *filter_id);

  RESPONSE filter_collect();
  RESPONSE filter_collect_mthread();

  // assign model settings to filters and initialize their model variable
  RESPONSE filter_allocate(bool to_assign = true);

  RESPONSE filter_train();
  RESPONSE filter_train_mthread();

  std::reference_wrapper<Config> config_;

  std::unique_ptr<BufferManager> buffer_manager_;

  std::unique_ptr<Node> root_;
  ID_TYPE nnode_, nleaf_;
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare> leaf_min_heap_;

  std::unique_ptr<Allocator> allocator_;

  VALUE_TYPE *filter_train_query_ptr_;
  torch::Tensor filter_train_query_tsr_;
  torch::Tensor filter_query_tsr_;
  std::unique_ptr<torch::Device> device_;
  std::stack<std::reference_wrapper<Filter>> filter_cache_;
};

}
}

#endif //DSTREE_INDEX_H
