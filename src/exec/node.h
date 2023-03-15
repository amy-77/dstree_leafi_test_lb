//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_NODE_H
#define DSTREE_NODE_H

#include <memory>

#include <torch/torch.h>

#include "buffer.h"
#include "global.h"
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
       dstree::BufferManager &buffer_manager,
       ID_TYPE depth,
       ID_TYPE id);
  Node(Config &config,
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
    return eapca_envelope_->cal_lower_bound_EDsquare(series_ptr);
  }

  bool has_filter() const { return filter_ != nullptr; }
  bool has_active_filter() const { return filter_ != nullptr && filter_->is_active(); }
  Filter &get_filter() const { return *filter_; }
  VALUE_TYPE filter_infer(torch::Tensor &query_series) const { return filter_->infer(query_series); }

  VALUE_TYPE get_envelop_pruning_frequency() const {
    return filter_.get()->get_external_pruning_frequency();
  };

  RESPONSE implant_filter(ID_TYPE id, std::reference_wrapper<torch::Tensor> shared_train_queries) {
    filter_ = std::make_unique<dstree::Filter>(config_, id, shared_train_queries);
    return SUCCESS;
  }

  RESPONSE activate_filter(const MODEL_SETTING &model_setting) {
    if (filter_ != nullptr) {
      return filter_->activate(model_setting);
    } else {
      return FAILURE;
    }
  }

  RESPONSE push_filter_example(VALUE_TYPE bsf_distance, VALUE_TYPE nn_distance, VALUE_TYPE lb_distance) {
    return filter_->push_example(bsf_distance, nn_distance, lb_distance);
  }

  std::vector<std::reference_wrapper<Node>>::iterator begin() {
    if (children_refs_.empty()) {
      for (auto &child_node : children_) {
        children_refs_.push_back(std::ref(*child_node));
      }
    }
    return children_refs_.begin();
  }

  std::vector<std::reference_wrapper<Node>>::iterator end() {
    if (children_refs_.empty()) {
      for (auto &child_node : children_) {
        children_refs_.push_back(std::ref(*child_node));
      }
    }
    return children_refs_.end();
  }

  ID_TYPE get_id() const { return id_; }
  ID_TYPE get_size() const { return nseries_; }

  bool is_full() const { return nseries_ == config_.get().leaf_max_nseries_; }
  bool is_leaf() const { return children_.empty(); }

  RESPONSE log();

  RESPONSE dump(void *ofs_buf) const;
  RESPONSE load(void *ifs_buf,
                dstree::BufferManager &buffer_manager,
                ID_TYPE &nnode,
                ID_TYPE &nleaf);

 private:
  ID_TYPE depth_, id_;
  ID_TYPE nseries_;

  std::reference_wrapper<Config> config_;

  std::reference_wrapper<Buffer> buffer_;

  std::unique_ptr<EAPCAEnvelope> eapca_envelope_;
  std::unique_ptr<Split> split_;

  std::vector<std::unique_ptr<Node>> children_;
  std::vector<std::reference_wrapper<Node>> children_refs_; // for iterator only

  std::unique_ptr<Filter> filter_;
};

}
}

#endif //DSTREE_NODE_H
