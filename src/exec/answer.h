//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_QUERYANSWER_H
#define DSTREE_QUERYANSWER_H

#include <queue>
#include <functional>

#include "global.h"

namespace upcite {
namespace dstree {

class Answer {
 public:
  explicit Answer(ID_TYPE query_id);
  ~Answer() = default;

 private:
  ID_TYPE query_id_;

  std::priority_queue<VALUE_TYPE, std::vector<VALUE_TYPE>, std::less<>> bsfs_;
};

}
}

#endif //DSTREE_QUERYANSWER_H
