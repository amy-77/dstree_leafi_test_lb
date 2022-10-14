//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_GLOBAL_H
#define DSTREE_GLOBAL_H

#define DEBUG
#define DEBUGGED

typedef float VALUE_TYPE;
typedef int64_t ID_TYPE;

enum RESPONSE {
  SUCCESS = 0,
  FAILURE = 1
};

namespace upcite {
namespace constant {

const VALUE_TYPE EPSILON = 1e-5;

const VALUE_TYPE MIN_VALUE = -1e5;
const VALUE_TYPE MAX_VALUE = 1e5;

}
}

#endif //DSTREE_GLOBAL_H
