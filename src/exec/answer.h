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
  Answer(ID_TYPE capacity, ID_TYPE query_id);
  ~Answer() = default;

  bool is_bsf(VALUE_TYPE distance) const {
    if (bsf_distances_.size() < capacity_) {
      return true;
    } else {
      return distance < bsf_distances_.top();
    }
  }

  RESPONSE push_bsf(VALUE_TYPE distance);
  RESPONSE check_push_bsf(VALUE_TYPE distance);

  VALUE_TYPE get_bsf() const { return bsf_distances_.top(); };
  VALUE_TYPE pop_bsf();

  RESPONSE reset(ID_TYPE query_id) {
    query_id_ = query_id;

    while (!bsf_distances_.empty()) {
      bsf_distances_.pop();
    }

    return SUCCESS;
  }

  bool empty() const { return bsf_distances_.empty(); }

  ID_TYPE query_id_;

  std::priority_queue<VALUE_TYPE, std::vector<VALUE_TYPE>, std::less<>> bsf_distances_;

 private:
  ID_TYPE capacity_;

  // TODO with-id bsf heap
//  std::priority_queue<VALUE_TYPE, std::vector<VALUE_TYPE>, std::greater<>> bsf_distances_;
};

}
}

#endif //DSTREE_QUERYANSWER_H
