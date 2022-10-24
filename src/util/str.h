//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_UTIL_STR_H_
#define DSTREE_SRC_UTIL_STR_H_

#include <string>

#include "global.h"

namespace upcite {

template<class T>
static std::string get_str(T *values, ID_TYPE length) {
  return std::accumulate(values + 1,
                         values + length,
                         std::to_string(values[0]),
                         [](const std::string &a, T b) {
                           return a + ", " + std::to_string(b);
                         });
}

}

#endif //DSTREE_SRC_UTIL_STR_H_
