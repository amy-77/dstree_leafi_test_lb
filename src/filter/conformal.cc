//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "conformal.h"

#include <algorithm>

#include "spdlog/spdlog.h"

#include "comp.h"
#include "vec.h"

upcite::ConformalRegressor::ConformalRegressor(std::string core_type_str,
                                               VALUE_TYPE confidence) :
    gsl_accel_(nullptr),
    gsl_spline_(nullptr) {
  if (core_type_str == "discrete") {
    core_ = DISCRETE;
  } else if (core_type_str == "spline") {
    core_ = SPLINE;
  } else {
    spdlog::error("conformal core {:s} is not recognized; roll back to the default: discrete",
                  core_type_str);
    core_ = DISCRETE;
  }

  confidence_level_ = confidence;
}

RESPONSE upcite::ConformalRegressor::fit(std::vector<ERROR_TYPE> &residuals) {
  alphas_.assign(residuals.begin(), residuals.end());
  for (auto &residual : alphas_) { residual = residual < 0 ? -residual : residual; }

  std::sort(alphas_.begin(), alphas_.end()); //non-decreasing

  if (core_ == DISCRETE) {
    is_fitted_ = true;
    is_trial_ = false;

    abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level_);
    alpha_ = alphas_[abs_error_i_];
  } else { // core_ == SPLINE
    // fit later with recalls as input
    is_fitted_ = false;
  }

  return SUCCESS;
}

RESPONSE upcite::ConformalRegressor::fit_spline(std::string &spline_core, std::vector<ERROR_TYPE> &recalls) {
  assert(recalls.size() == alphas_.size());
  gsl_accel_ = std::unique_ptr<gsl_interp_accel>(gsl_interp_accel_alloc());

  if (spline_core == "steffen") {
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  } else if (spline_core == "cubic") {
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_cspline, recalls.size()));
  } else {
    spdlog::error("conformal spline core {:s} is not recognized; roll back to the default: steffen", spline_core);

    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  }

  gsl_spline_init(gsl_spline_.get(), recalls.data(), alphas_.data(), recalls.size());

  is_fitted_ = true;
  is_trial_ = false;

  return SUCCESS;
}

upcite::INTERVAL upcite::ConformalRegressor::predict(VALUE_TYPE y_hat,
                                                     VALUE_TYPE confidence_level,
                                                     VALUE_TYPE y_max,
                                                     VALUE_TYPE y_min) {
  if (is_fitted_) {
    if (confidence_level >= 0 && !upcite::is_equal(confidence_level_, confidence_level)) {
      abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level);
      alpha_ = alphas_[abs_error_i_];
      confidence_level_ = confidence_level;
    }

    return {y_hat - alpha_, y_hat + alpha_};
  } else if (is_trial_) {
    return {y_hat - alpha_, y_hat + alpha_};
  } else {
    return {y_min, y_max};
  }
}

RESPONSE upcite::ConformalPredictor::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&core_), sizeof(CONFORMAL_CORE));

  // alphas can be recalculated
//  ID_TYPE alphas_size = static_cast<ID_TYPE>(alphas_.size());
//  node_fos.write(reinterpret_cast<const char *>(&alphas_size), sizeof(ID_TYPE));
//  node_fos.write(reinterpret_cast<const char *>(alphas_.data()), sizeof(VALUE_TYPE) * alphas_.size());

  return SUCCESS;
}

RESPONSE upcite::ConformalPredictor::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_core_buf = reinterpret_cast<CONFORMAL_CORE *>(ifs_buf);
//  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
//  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(CONFORMAL_CORE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  core_ = ifs_core_buf[0];

//  read_nbytes = sizeof(ID_TYPE);
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  ID_TYPE alphas_size = ifs_id_buf[0];
//  alphas_.reserve(alphas_size);
//
//  read_nbytes = sizeof(VALUE_TYPE) * alphas_size;
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  alphas_.insert(alphas_.begin(), ifs_value_buf, ifs_value_buf + alphas_size);

  return SUCCESS;
}

VALUE_TYPE upcite::ConformalPredictor::get_alpha() const {
  if (is_fitted_) {
    return alpha_;
  }

  return constant::MAX_VALUE;
}

RESPONSE upcite::ConformalPredictor::set_alpha(VALUE_TYPE alpha, bool is_trial) {
  if (is_trial) {
    if (is_fitted_) {
      spdlog::error("conformal model is already fitted; cannot run trial");
      return FAILURE;
    } else {
      alpha_ = alpha;

      is_trial_ = true;
    }
  } else if (is_fitted_) {
    spdlog::error("conformal model is already fitted; cannot directly adjust alpha");
    return FAILURE;
  } else {
    alpha_ = alpha;
  }

  return SUCCESS;
}
VALUE_TYPE upcite::ConformalPredictor::get_alpha_by_pos(ID_TYPE pos) const {
  if (pos >= 0 && pos < alphas_.size()) {
    return alphas_[pos];
  }

  return constant::MAX_VALUE;
}

RESPONSE upcite::ConformalPredictor::set_alpha_by_pos(ID_TYPE pos) {
  if (pos >= 0 && pos < alphas_.size()) {
    alpha_ = alphas_[pos];
    confidence_level_ = -1;

    return SUCCESS;
  }

  return FAILURE;
}

RESPONSE upcite::ConformalRegressor::set_alpha_by_recall(VALUE_TYPE recall) {
  assert(gsl_accel_ != nullptr && gsl_spline_ != nullptr);

  alpha_ = gsl_spline_eval(gsl_spline_.get(), recall, gsl_accel_.get());
  confidence_level_ = -1;

  return SUCCESS;
}
