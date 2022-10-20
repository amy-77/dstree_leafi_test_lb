//
// Created by Qitong Wang on 2022/10/19.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_DISTANCE_H_
#define DSTREE_SRC_EXEC_DISTANCE_H_

#include "global.h"

namespace upcite {

static VALUE_TYPE cal_EDsquare(const VALUE_TYPE *series_1_ptr,
                               const VALUE_TYPE *series_2_ptr,
                               ID_TYPE series_length) {
  VALUE_TYPE distance = 0;

  while (series_length > 0) {
    distance += (series_1_ptr[series_length] - series_2_ptr[series_length])
        * (series_1_ptr[series_length] - series_2_ptr[series_length]);

    series_length--;
  }

  //return sqrtf(distance);
  return distance;
}

}

#endif //DSTREE_SRC_EXEC_DISTANCE_H_
