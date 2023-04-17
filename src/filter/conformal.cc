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
                                               VALUE_TYPE confidence) {
  if (core_type_str == "histogram") {
    core_ = HISTOGRAM;
  } else if (core_type_str == "mlp") {
    spdlog::error("conformal core {:s} is not implemented; roll back to the default HISTOGRAM",
                  core_type_str);
//    core_ = MLMODEL;
    core_ = HISTOGRAM;
  } else {
    spdlog::error("conformal core {:s} is not recognized; roll back to the default HISTOGRAM",
                  core_type_str);
  }

  confidence_ = confidence;
}

RESPONSE upcite::ConformalRegressor::fit(std::vector<VALUE_TYPE> &residuals) {
  alphas_.assign(residuals.begin(), residuals.end());
  for (auto &residual : alphas_) { residual = residual < 0 ? -residual : residual; }

  std::sort(alphas_.begin(), alphas_.end()); //non-decreasing

  is_fitted_ = true;

  confidence_id_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_);
  alpha_ = alphas_[confidence_id_];

  return SUCCESS;
}

upcite::INTERVAL upcite::ConformalRegressor::predict(VALUE_TYPE y_hat,
                                                     VALUE_TYPE confidence,
                                                     VALUE_TYPE y_max,
                                                     VALUE_TYPE y_min) {
  if (is_fitted_) {
    if (confidence >= 0 && !upcite::is_equal(confidence_, confidence)) {
      confidence_id_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence);
      alpha_ = alphas_[confidence_id_];
      confidence_ = confidence;
    }

    return {y_hat - alpha_, y_hat + alpha_};
  } else {
    return {y_min, y_max};
  }
}

std::vector<upcite::INTERVAL> upcite::ConformalRegressor::predict(std::vector<VALUE_TYPE> &y_hat,
                                                                  VALUE_TYPE confidence,
                                                                  VALUE_TYPE y_max,
                                                                  VALUE_TYPE y_min) {
  if (confidence >= 0 && confidence <= 1 && !upcite::is_equal(confidence_, confidence)) {
    confidence_id_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence);
    alpha_ = alphas_[confidence_id_];
    confidence_ = confidence;
  }

  auto y_intervals = upcite::make_reserved<upcite::INTERVAL>(y_hat.size());

  for (VALUE_TYPE y_i : y_hat) {
    y_intervals.emplace_back(y_i - alpha_, y_i + alpha_);
  }

  return y_intervals;
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

VALUE_TYPE upcite::ConformalPredictor::get_alpha(VALUE_TYPE confidence) const {
  if (is_fitted_) {
    if (confidence >= 0 && confidence <= 1) {
      if (upcite::is_equal(confidence_, confidence)) {
        return alpha_;
      } else {
        ID_TYPE confidence_id = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence);
        return alphas_[confidence_id];
      }
    }
  }

  return constant::MAX_VALUE;
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
    return SUCCESS;
  }

  return FAILURE;
}
