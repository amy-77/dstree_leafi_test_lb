//
// Created by Qitong Wang on 2022/10/10.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "split.h"

namespace dstree = upcite::dstree;

ID_TYPE dstree::Split::route(VALUE_TYPE value) const {
  auto horizontal_segment_id = static_cast<ID_TYPE>(horizontal_breakpoints_.size());

  for (ID_TYPE i = 0; i < horizontal_breakpoints_.size(); ++i) {
    if (value < horizontal_breakpoints_[i]) {
      horizontal_segment_id = i;
    }
  }

  return horizontal_segment_id;
}

dstree::Split &dstree::Split::operator=(const dstree::Split &split) {
  is_vertical_split_ = split.is_vertical_split_;

  split_segment_id_ = split.split_segment_id_;
  split_subsegment_id_ = split.split_subsegment_id_;

  horizontal_split_mode_ = split.horizontal_split_mode_;
  horizontal_breakpoints_ = split.horizontal_breakpoints_;

  return *this;
}
