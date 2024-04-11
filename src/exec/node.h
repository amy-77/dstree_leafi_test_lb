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
  Node &route(dstree::EAPCA &series_eapca, bool is_update_statistics);

  RESPONSE enqueue_leaf(std::vector<std::reference_wrapper<Node>> &leaves);
//  RESPONSE enqueue_children(std::vector<std::shared_ptr<Node>> &leaves);

  RESPONSE insert(ID_TYPE series_id,
                  dstree::EAPCA &series_eapca);

  RESPONSE split(dstree::BufferManager &buffer_manager,
                 ID_TYPE first_child_id);

  RESPONSE search(const VALUE_TYPE *query_series_ptr,
                  Answers &answer,
                  ID_TYPE &visited_node_counter,
                  ID_TYPE &visited_series_counter) const;

  VALUE_TYPE search(const VALUE_TYPE *query_series_ptr,
                    VALUE_TYPE *m256_fetch_cache = nullptr,
                    VALUE_TYPE bsf_distance = -1) const;

  VALUE_TYPE search_mt(const VALUE_TYPE *query_series_ptr,
                       Answers &answer,
                       pthread_mutex_t *answer_mutex_) const;

  VALUE_TYPE cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr) const {
    return eapca_envelope_->cal_lower_bound_EDsquare(series_ptr);
  }

  bool has_filter() const { return filter_ != nullptr; }
  bool has_active_filter() const { return filter_ != nullptr && filter_->is_active(); }
  bool has_trained_filter() const { return filter_ != nullptr && filter_->is_trained(); }
  VALUE_TYPE filter_infer(torch::Tensor &query_series) const { return filter_->infer(query_series); }

  VALUE_TYPE get_filter_nn_distance(ID_TYPE pos) const { return filter_->get_nn_distance(pos); };
  VALUE_TYPE get_filter_bsf_distance(ID_TYPE pos) const { return filter_->get_bsf_distance(pos); };
  VALUE_TYPE get_filter_pred_distance(ID_TYPE pos) const { return filter_->get_pred_distance(pos); };
  std::tuple<VALUE_TYPE, VALUE_TYPE> get_filter_global_lnn_mean_std() const {
    return filter_->get_global_lnn_mean_std();
  };

  // TODO deprecate the object wrapper
  std::reference_wrapper<Filter> get_filter() { return std::ref(*filter_); }

  VALUE_TYPE get_filter_abs_error_interval() const {
    return filter_->get_abs_error_interval();
  };

  VALUE_TYPE get_filter_abs_error_interval_by_pos(ID_TYPE pos) const {
    return filter_->get_abs_error_interval_by_pos(pos);
  };
  RESPONSE set_filter_abs_error_interval_by_pos(ID_TYPE pos) {
    return filter_->set_abs_error_interval_by_pos(pos);
  };

  RESPONSE fit_filter_conformal_spline(std::vector<ERROR_TYPE> &recalls) {
    return filter_->fit_filter_conformal_spline(recalls);
  }

  RESPONSE set_filter_abs_error_interval_by_recall(VALUE_TYPE recall) {
    return filter_->set_abs_error_interval_by_recall(recall);
  };

  RESPONSE set_filter_abs_error_interval(VALUE_TYPE abs_error) {
    return filter_->set_abs_error_interval(abs_error);
  };

  VALUE_TYPE get_envelop_pruning_frequency() const {
    return filter_.get()->get_node_summarization_pruning_frequency();
  };

  RESPONSE add_filter(ID_TYPE id, std::reference_wrapper<torch::Tensor> shared_train_queries) {
    filter_ = std::make_unique<dstree::Filter>(config_, id, shared_train_queries);
    return SUCCESS;
  }

  RESPONSE activate_filter(MODEL_SETTING &model_setting) {
    if (filter_ != nullptr) {
      return filter_->activate(model_setting);
    } else {
      return FAILURE;
    }
  }

  RESPONSE deactivate_filter() {
    if (filter_ != nullptr) {
      return filter_->deactivate();
    } else {
      return FAILURE;
    }
  }

  RESPONSE push_global_example(VALUE_TYPE bsf_distance, VALUE_TYPE nn_distance, VALUE_TYPE lb_distance) {
    return filter_->push_global_example(bsf_distance, nn_distance, lb_distance);
  }
  RESPONSE push_local_example(VALUE_TYPE const *series, VALUE_TYPE nn_distance) {
    return filter_->push_local_example(series, nn_distance);
  }

//  RESPONSE dump_local_example() {
//    std::string filter_query_filepath = config_.get().index_dump_folderpath_ +
//        std::to_string(id_) + config_.get().filter_query_filename_;
//    return filter_->dump_local_example(filter_query_filepath);
//  }

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

  ID_TYPE get_num_synthetic_queries(ID_TYPE node_size_threshold);
  RESPONSE synthesize_query(VALUE_TYPE *generated_queries, ID_TYPE &num_generated_queries, ID_TYPE node_size_threshold);

  VALUE_TYPE const *get_series_ptr_by_id(ID_TYPE series_id) { return buffer_.get().get_series_ptr_by_id(series_id); }

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
