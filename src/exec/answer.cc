//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "answer.h"

#include "vec.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Answers::Answers(ID_TYPE capacity, ID_TYPE query_id) :
    capacity_(capacity),
    query_id_(query_id),
    bsf_distance_(constant::MAX_VALUE) {
//  std::priority_queue<Answer, std::vector<Answer>, compAnswerLess> bsf_distances_;
  bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(
      compAnswerLess(), make_reserved<Answer>(capacity + 1));
}

RESPONSE dstree::Answers::push_bsf(VALUE_TYPE distance, ID_TYPE node_id) {
  bsf_distances_.emplace(distance, node_id);

  if (bsf_distances_.size() > capacity_) {
    bsf_distances_.pop();
  }

  bsf_distance_ = bsf_distances_.top().nn_dist_;

  return SUCCESS;
}

RESPONSE dstree::Answers::check_push_bsf(VALUE_TYPE distance, ID_TYPE node_id) {
  if (is_bsf(distance)) {
    push_bsf(distance, node_id);
  }

  return SUCCESS;
}

VALUE_TYPE dstree::Answers::pop_bsf() {
  VALUE_TYPE bsf = bsf_distances_.top().nn_dist_;
  bsf_distances_.pop();

  return bsf;
}

upcite::Answer dstree::Answers::pop_answer() {
  auto answer = bsf_distances_.top();
  bsf_distances_.pop();

  return answer;
}
