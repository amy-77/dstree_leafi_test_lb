//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_FILTER_H
#define DSTREE_FILTER_H

#include <memory>
#include <functional>

#include <torch/torch.h>
#include <spdlog/spdlog.h>

#include "global.h"
#include "config.h"
#include "conformal.h"
#include "model.h"

namespace upcite {
namespace dstree {

class Filter {
 public:
  Filter(dstree::Config &config,
         ID_TYPE id,
         std::reference_wrapper<torch::Tensor> shared_train_queries);
  ~Filter() = default;

  RESPONSE push_example(VALUE_TYPE bsf_distance, VALUE_TYPE nn_distance, VALUE_TYPE lb_distance) {
    bsf_distances_.push_back(bsf_distance);
    nn_distances_.push_back(nn_distance);

    lb_distances_.push_back(lb_distance);

    train_size_ += 1;

    return SUCCESS;
  };

  bool is_active() const { return is_active_; }
  RESPONSE activate(MODEL_SETTING &model_setting) {
    model_setting_ref_ = model_setting;
    is_active_ = true;

    return SUCCESS;
  }

  RESPONSE trigger_trial(MODEL_SETTING &model_setting) {
    model_setting_ref_ = model_setting;
    is_active_ = false;

    return SUCCESS;
  }

  RESPONSE train(bool is_trial = false);
  VALUE_TYPE infer(torch::Tensor &query_series) const;

  RESPONSE dump(std::ofstream &node_fos) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

  ID_TYPE get_id() const { return id_; };
  VALUE_TYPE get_node_summarization_pruning_frequency() const;
  VALUE_TYPE get_nn_distance(ID_TYPE pos) const { return nn_distances_[pos]; };
  VALUE_TYPE get_bsf_distance(ID_TYPE pos) const { return bsf_distances_[pos]; };
  VALUE_TYPE get_pred_distance(ID_TYPE pos) const { return pred_distances_[pos]; };

  VALUE_TYPE get_val_pruning_ratio() const;

  VALUE_TYPE get_abs_error_interval() const {
    return conformal_predictor_->get_alpha();
  };

  VALUE_TYPE get_abs_error_interval_by_pos(ID_TYPE pos) const {
    return conformal_predictor_->get_alpha_by_pos(pos);
  };
  RESPONSE set_abs_error_interval_by_pos(ID_TYPE pos) {
    return conformal_predictor_->set_alpha_by_pos(pos);
  };

  RESPONSE fit_filter_conformal_spline(std::vector<ERROR_TYPE> &recalls) {
    return conformal_predictor_->fit_spline(config_.get().filter_conformal_smoothen_core_, recalls);
  }

  RESPONSE set_abs_error_interval_by_recall(VALUE_TYPE recall) {
    return conformal_predictor_->set_alpha_by_recall(recall);
  };

 private:
  RESPONSE fit_conformal_predictor(bool is_trial = false);

  std::reference_wrapper<dstree::Config> config_;

  ID_TYPE id_;

  bool is_active_;
  std::reference_wrapper<MODEL_SETTING> model_setting_ref_;
  // for filter loading only
  MODEL_SETTING model_setting_;
  // torch::save only takes shared_ptr
  std::shared_ptr<FilterModel> model_;
  std::unique_ptr<ConformalRegressor> conformal_predictor_;

  // TODO ref?
  // TODO support different device for training and inference
  std::unique_ptr<torch::Device> device_;

  bool is_trained_;
  bool is_distances_preprocessed_, is_distances_logged;
  ID_TYPE train_size_;

  std::reference_wrapper<torch::Tensor> shared_train_queries_;
  std::vector<VALUE_TYPE> bsf_distances_;
  std::vector<VALUE_TYPE> nn_distances_;

  std::vector<VALUE_TYPE> lb_distances_; // lower bounds
  std::vector<VALUE_TYPE> ub_distances_; // upper bounds

  std::vector<VALUE_TYPE> pred_distances_;
};

}
}

#endif //DSTREE_FILTER_H
