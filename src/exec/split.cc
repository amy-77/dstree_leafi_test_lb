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

RESPONSE dstree::Split::dump(std::ofstream &node_fos, void *ofs_buf) const {
  node_fos.write(reinterpret_cast<const char *>(&is_vertical_split_), sizeof(bool));

  node_fos.write(reinterpret_cast<const char *>(&split_segment_id_), sizeof(ID_TYPE));
  node_fos.write(reinterpret_cast<const char *>(&split_subsegment_id_), sizeof(ID_TYPE));
  node_fos.write(reinterpret_cast<const char *>(&split_segment_offset_), sizeof(ID_TYPE));
  node_fos.write(reinterpret_cast<const char *>(&split_segment_length_), sizeof(ID_TYPE));

  node_fos.write(reinterpret_cast<const char *>(&horizontal_split_mode_), sizeof(HORIZONTAL_SPLIT_MODE));

  auto ofs_id_buf = reinterpret_cast<ID_TYPE *>(ofs_buf);
  ofs_id_buf[0] = static_cast<ID_TYPE>(horizontal_breakpoints_.size());
  node_fos.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));
  node_fos.write(reinterpret_cast<const char *>(horizontal_breakpoints_.data()),
                 sizeof(VALUE_TYPE) * horizontal_breakpoints_.size());

  return SUCCESS;
}
