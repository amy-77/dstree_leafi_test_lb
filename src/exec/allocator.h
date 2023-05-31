//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_ALLOCATOR_H_
#define DSTREE_SRC_EXEC_ALLOCATOR_H_

#include <vector>
#include <map>
#include <tuple>

#include "global.h"
#include "config.h"
#include "filter_core.h"
#include "node.h"

namespace upcite {
namespace dstree {

struct FilterInfo {
 public:
  explicit FilterInfo(Node &node) :
      node_(node),
      model_setting(upcite::MODEL_SETTING_PLACEHOLDER_REF) {
    score = -1;

    external_pruning_probability_ = -1;
  };
  ~FilterInfo() = default;

  VALUE_TYPE score;
  std::reference_wrapper<MODEL_SETTING> model_setting;

  std::reference_wrapper<Node> node_;

  VALUE_TYPE external_pruning_probability_; // d_bsf < d_lb
//  VALUE_TYPE pruning_probability_; // d_bsf < d_p
//  VALUE_TYPE false_pruning_probability_; // d_nn < d_bsf < d_p
};

static bool compDecreFilterScore(dstree::FilterInfo &filter_info_1, dstree::FilterInfo &filter_info_2) {
  return filter_info_1.score > filter_info_2.score;
}

static bool compDecreFilterNSeries(dstree::FilterInfo &filter_info_1, dstree::FilterInfo &filter_info_2) {
  return filter_info_1.node_.get().get_size() > filter_info_2.node_.get().get_size();
}

class Allocator {
 public:
  explicit Allocator(Config &config,
                     ID_TYPE nfilters = -1);
  ~Allocator() = default;

  RESPONSE push_instance(const FilterInfo &filter_info);

  RESPONSE assign();

  RESPONSE set_confidence_from_recall();

 private:
  RESPONSE trial_collect_mthread();
  RESPONSE evaluate();

  std::reference_wrapper<Config> config_;

  double_t cpu_ms_per_series_;

  std::vector<MODEL_SETTING> candidate_model_settings_;
  VALUE_TYPE available_gpu_memory_mb_;

  std::vector<FilterInfo> filter_infos_;

  bool is_recall_calculated_;
  std::vector<ERROR_TYPE> validation_recalls_;

  std::vector<ID_TYPE> filter_ids_;
  std::vector<VALUE_TYPE> gains_matrix_;
  std::vector<VALUE_TYPE> mem_matrix_;

  // TODO memory is continuous instead of discrete
//  std::unique_ptr<VALUE_TYPE> total_gain_matrix_;
//  std::unique_ptr<bool> path_matrix_;
  std::vector<std::vector<std::tuple<VALUE_TYPE, VALUE_TYPE>>> total_gain_matrix_;
  std::vector<std::vector<ID_TYPE>> path_matrix_;
};

} // namespace dstree
} // namespace upcite

#endif //DSTREE_SRC_EXEC_ALLOCATOR_H_
