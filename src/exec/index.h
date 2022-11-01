//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_INDEX_H
#define DSTREE_INDEX_H

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "global.h"
#include "config.h"
#include "buffer.h"
#include "node.h"
#include "filter.h"

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
  ~Index();

  RESPONSE build();
  RESPONSE train();

  RESPONSE load();
  RESPONSE dump();

  RESPONSE search();
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *series_ptr);

 private:
  RESPONSE insert(ID_TYPE batch_series_id);

  RESPONSE nf_initialize(std::shared_ptr<dstree::Node> &node,
                         ID_TYPE *filter_id);
  RESPONSE nf_collect();
  RESPONSE nf_collect_mthread();
  RESPONSE nf_train();

  std::shared_ptr<Config> config_;
  std::shared_ptr<upcite::Logger> logger_;

  std::unique_ptr<BufferManager> buffer_manager_;

  std::shared_ptr<Node> root_;
  ID_TYPE nnode_, nleaf_;
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, Compare> leaf_min_heap_;

  VALUE_TYPE *nf_train_query_ptr_;
  torch::Tensor nf_train_query_tsr_;
  torch::Tensor nf_query_tsr_;
  std::unique_ptr<torch::Device> device_;
  std::vector<std::reference_wrapper<Filter>> filter_cache_;
};

}
}

#endif //DSTREE_INDEX_H
