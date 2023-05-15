//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_ALLOCATOR_H_
#define DSTREE_SRC_EXEC_ALLOCATOR_H_

#include <vector>
#include <map>

#include "global.h"
#include "config.h"
#include "model.h"
#include "node.h"

namespace upcite {
namespace dstree {

struct FilterInfo {
 public:
  explicit FilterInfo(Node &node) :
      node_(node) {
    score = -1;
    model_setting = MODEL_SETTING();

    external_pruning_probability_ = -1;
    pruning_probability_ = -1;
    false_pruning_probability_ = -1;
  };
  ~FilterInfo() = default;

  VALUE_TYPE score;
  MODEL_SETTING model_setting;

  std::reference_wrapper<Node> node_;

  // TODO collect these stats
  VALUE_TYPE external_pruning_probability_; // d_bsf < d_lb
  // TODO diff: d_bsf < d_f when d_bsf >= d_lb vs. d_bsf < d_f
  VALUE_TYPE pruning_probability_; // d_bsf < d_p
  // TODO how to use false_pruning_probability_?
  VALUE_TYPE false_pruning_probability_; // d_nn < d_bsf < d_p
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

  RESPONSE push_instance(const FilterInfo& filter_info);

  // TODO this is a combination optimization problem
  // the current solution is to find the best model setting for each filter, and then pick the top filters
  // ideally one filter with a slower but more accurate model might be replaced by two filters with faster but less accurate models
  RESPONSE assign();

  RESPONSE set_confidence_from_recall();

 private:
  RESPONSE evaluate();

  std::reference_wrapper<Config> config_;

  // VALUE_TYPE num. series processed per second
  // TODO test cpu_sps_
  VALUE_TYPE cpu_sps_;

  // overhead per node: extra compared to each other
  // cpu overhead: none
  // gpu overhead: exchange between gpu memory and main memory; fixed across models
  // TODO test gpu_overhead_
  VALUE_TYPE cpu_overhead_pn_, gpu_overhead_pn_;

  // however performance of different combinations should be provided (or tested?)
  // maybe select a subset of (e.g., 10) nodes, train all candidate models on these
  // nodes, check (gpu) memory footprint, test speed and estimate accuracy
  // number of nodes ~10^4
  // number of trail trains: (#threads, or 16, say) * ~10 (should be a selected subset of all possible combinations)
  std::vector<MODEL_SETTING> candidate_model_settings_;
  std::map<std::string, VALUE_TYPE> model_gpu_memory_fingerprint_mb_;
  VALUE_TYPE available_gpu_memory_mb_;

  std::vector<FilterInfo> filter_infos_;

  bool is_recall_calculated_;
  std::vector<ERROR_TYPE> validation_recalls_;
};

} // namespace dstree
} // namespace upcite

#endif //DSTREE_SRC_EXEC_ALLOCATOR_H_
