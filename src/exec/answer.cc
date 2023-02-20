//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "answer.h"

#include "vec.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Answer::Answer(ID_TYPE capacity, ID_TYPE query_id) :
    capacity_(capacity),
    query_id_(query_id),
    bsf_distance_(constant::MAX_VALUE) {
  bsf_distances_ = std::priority_queue<VALUE_TYPE, std::vector<VALUE_TYPE>, std::less<>>(
      std::less<>(), make_reserved<VALUE_TYPE>(capacity + 1));
}

RESPONSE dstree::Answer::push_bsf(VALUE_TYPE distance) {
  bsf_distances_.push(distance);

  if (bsf_distances_.size() > capacity_) {
    bsf_distances_.pop();
  }

  bsf_distance_ = bsf_distances_.top();

  return SUCCESS;
}

RESPONSE dstree::Answer::check_push_bsf(VALUE_TYPE distance) {
  if (is_bsf(distance)) {
    push_bsf(distance);
  }

  return SUCCESS;
}

VALUE_TYPE dstree::Answer::pop_bsf() {
  VALUE_TYPE bsf = bsf_distances_.top();
  bsf_distances_.pop();

  return bsf;
}
