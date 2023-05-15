//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
// credit: https://github.com/henrikbostrom/crepes
//

#ifndef DSTREE_SRC_FILTER_CONFORMAL_H_
#define DSTREE_SRC_FILTER_CONFORMAL_H_

#include <vector>

#include "global.h"
#include "interval.h"
#include <gsl/gsl_spline.h>

namespace upcite {

enum CONFORMAL_CORE {
  DISCRETE = 0,
  SPLINE = 1 // smoothened
};

class ConformalPredictor {
 public:
  ConformalPredictor() : is_fitted_(false), is_trial_(false) {};
  ~ConformalPredictor() = default;

  VALUE_TYPE get_alpha() const;
  RESPONSE set_alpha(VALUE_TYPE alpha, bool is_trial = true);

  VALUE_TYPE get_alpha_by_pos(ID_TYPE pos) const;
  RESPONSE set_alpha_by_pos(ID_TYPE pos);

  RESPONSE dump(std::ofstream &node_fos) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

 protected:
  bool is_fitted_;
  bool is_trial_;
  CONFORMAL_CORE core_;

  VALUE_TYPE confidence_level_;
  ID_TYPE abs_error_i_;
  VALUE_TYPE alpha_;

  std::vector<ERROR_TYPE> alphas_;
};

class ConformalRegressor : public ConformalPredictor {
 public:
  explicit ConformalRegressor(std::string core_type_str, VALUE_TYPE confidence);
  ~ConformalRegressor() = default;

  RESPONSE set_alpha_by_recall(VALUE_TYPE recall);

  RESPONSE fit(std::vector<ERROR_TYPE> &residuals);
//               std::vector<VALUE_TYPE> &sigmas, std::vector<ID_TYPE> &bins);

  RESPONSE fit_spline(std::string &spline_core, std::vector<ERROR_TYPE> &recalls);

  INTERVAL predict(VALUE_TYPE y_hat,
                   VALUE_TYPE confidence_level = -1,
                   VALUE_TYPE y_max = constant::MAX_VALUE,
                   VALUE_TYPE y_min = constant::MIN_VALUE);

 private:
  std::unique_ptr<gsl_interp_accel> gsl_accel_;
  std::unique_ptr<gsl_spline> gsl_spline_;
};

}

#endif //DSTREE_SRC_FILTER_CONFORMAL_H_
