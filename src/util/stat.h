//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_STAT_H_
#define DSTREE_SRC_EXEC_STAT_H_

#include <tuple>
#include <cmath>

#include "global.h"

namespace upcite {

static VALUE_TYPE cal_mean(const VALUE_TYPE *series_ptr,
                           ID_TYPE nvalues) {
  VALUE_TYPE mean = 0;

  for (auto i = 0; i < nvalues; ++i) {
    mean += series_ptr[i];
  }
  mean /= static_cast<VALUE_TYPE>(nvalues);

  return mean;
}

static VALUE_TYPE cal_std(const VALUE_TYPE *series_ptr,
                          ID_TYPE nvalues) {
  VALUE_TYPE mean = 0, square_sum = 0;

  for (auto i = 0; i < nvalues; ++i) {
    mean += series_ptr[i];
  }
  mean /= static_cast<VALUE_TYPE>(nvalues);

  for (auto i = 0; i < nvalues; ++i) {
    square_sum += (series_ptr[i] - mean) * (series_ptr[i] - mean);
  }
  square_sum /= static_cast<VALUE_TYPE>(nvalues);

  return sqrt(square_sum);
}

static std::tuple<VALUE_TYPE, VALUE_TYPE> cal_mean_std(const VALUE_TYPE *series_ptr,
                                                       ID_TYPE nvalues) {
  VALUE_TYPE mean = 0, square_sum = 0;

  for (auto i = 0; i < nvalues; ++i) {
    mean += series_ptr[i];
  }
  mean /= static_cast<VALUE_TYPE>(nvalues);

  for (auto i = 0; i < nvalues; ++i) {
    square_sum += (series_ptr[i] - mean) * (series_ptr[i] - mean);
  }
  square_sum /= static_cast<VALUE_TYPE>(nvalues);

  return std::make_tuple(mean, sqrt(square_sum));
}

}

#endif //DSTREE_SRC_EXEC_STAT_H_
