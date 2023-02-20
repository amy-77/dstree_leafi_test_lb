//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_COMMON_COMP_H_
#define DSTREE_SRC_COMMON_COMP_H_

#include <limits>
#include <cmath>

namespace upcite {

template<typename T>
static bool is_equal(T arg_1, T arg_2) {
  return (std::fabs(arg_1 - arg_2) <= std::numeric_limits<T>::epsilon() * std::fmax(std::fabs(arg_1), std::fabs(arg_2)));
}

}

#endif //DSTREE_SRC_COMMON_COMP_H_
