//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
// credit: https://github.com/henrikbostrom/crepes
//

#ifndef DSTREE_SRC_FILTER_CONFORMAL_H_
#define DSTREE_SRC_FILTER_CONFORMAL_H_

#include <vector>

#include "global.h"
#include "intervel.h"

namespace upcite {

enum CONFORMAL_CORE {
  HISTOGRAM = 0,
  MLP = 1
};

class ConformalPredictor {
 public:
  ConformalPredictor() : is_fitted_(false) {};
  ~ConformalPredictor() = default;

  RESPONSE dump(std::ofstream &node_fos) const;
  RESPONSE load(std::ifstream &node_ifs, void *ifs_buf);

 protected:
  bool is_fitted_;
  CONFORMAL_CORE core_;

  VALUE_TYPE confidence_;
  ID_TYPE confidence_id_;
  VALUE_TYPE alpha_;

  std::vector<VALUE_TYPE> alphas_;
};

class ConformalRegressor : public ConformalPredictor {
 public:
  explicit ConformalRegressor(std::string core_type_str, VALUE_TYPE confidence);
  ~ConformalRegressor() = default;

  RESPONSE fit(std::vector<VALUE_TYPE> &residuals);
//               std::vector<VALUE_TYPE> &sigmas,
//               std::vector<ID_TYPE> &bins);

  INTERVAL predict(VALUE_TYPE y_hat,
                   VALUE_TYPE confidence = -1,
                   VALUE_TYPE y_max = constant::MAX_VALUE,
                   VALUE_TYPE y_min = constant::MIN_VALUE);

  std::vector<INTERVAL> predict(std::vector<VALUE_TYPE> &y_hat,
//                                  std::vector<VALUE_TYPE> &sigmas,
//                                  std::vector<ID_TYPE> &bins,
                                VALUE_TYPE confidence = -1,
                                VALUE_TYPE y_max = constant::MAX_VALUE,
                                VALUE_TYPE y_min = constant::MIN_VALUE);

};

}

#endif //DSTREE_SRC_FILTER_CONFORMAL_H_
