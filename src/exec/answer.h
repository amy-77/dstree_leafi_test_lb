//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_QUERYANSWER_H
#define DSTREE_QUERYANSWER_H

#include <queue>
#include <functional>

#include "global.h"

namespace constant = upcite::constant;

namespace upcite {

struct Answer {
 public:
  explicit Answer(VALUE_TYPE nn_dist, ID_TYPE node_id = -1) : nn_dist_(nn_dist), node_id_(node_id) {};
  ~Answer() = default;

  VALUE_TYPE nn_dist_;
  ID_TYPE node_id_;
};

struct compAnswerLess {
 public:
  bool operator()(Answer &answer_1, Answer &answer_2) const { return answer_1.nn_dist_ < answer_2.nn_dist_; }
};

namespace dstree {

class Answers {
 public:
  Answers(ID_TYPE capacity, ID_TYPE query_id);
  Answers(ID_TYPE capacity,
          ID_TYPE query_id,
          VALUE_TYPE bsf_distance,
          std::priority_queue<Answer, std::vector<Answer>, compAnswerLess> &bsf_distances) :
      capacity_(capacity),
      query_id_(query_id),
      bsf_distance_(bsf_distance),
      bsf_distances_(std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(bsf_distances)) {};
  ~Answers() = default;

  Answers(const Answers &answers) {
    capacity_ = answers.capacity_;
    query_id_ = answers.query_id_;
    bsf_distance_ = answers.bsf_distance_;
    bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(answers.bsf_distances_);
  }

  Answers &operator=(const Answers &answers) {
    capacity_ = answers.capacity_;
    query_id_ = answers.query_id_;
    bsf_distance_ = answers.bsf_distance_;
    bsf_distances_ = std::priority_queue<Answer, std::vector<Answer>, compAnswerLess>(answers.bsf_distances_);
    return *this;
  };

  bool is_bsf(VALUE_TYPE distance) const {
    if (bsf_distances_.size() < capacity_) {
      return true;
    } else {
//      return distance < bsf_distances_.top();
      return distance < bsf_distance_;
    }
  }

  RESPONSE push_bsf(VALUE_TYPE distance, ID_TYPE node_id = -1);
  RESPONSE check_push_bsf(VALUE_TYPE distance, ID_TYPE node_id = -1);

  VALUE_TYPE get_bsf() const {
//    return bsf_distances_.top();
    return bsf_distance_;
  };

  VALUE_TYPE pop_bsf();
  Answer pop_answer();

  RESPONSE reset(ID_TYPE query_id) {
    query_id_ = query_id;

    while (!bsf_distances_.empty()) {
      bsf_distances_.pop();
    }

    bsf_distance_ = constant::MAX_VALUE;

    return SUCCESS;
  }

  bool empty() const { return bsf_distances_.empty(); }

  ID_TYPE query_id_;

 private:
  ID_TYPE capacity_;
  VALUE_TYPE bsf_distance_;

  std::priority_queue<Answer, std::vector<Answer>, compAnswerLess> bsf_distances_;
};

}
}

#endif //DSTREE_QUERYANSWER_H
