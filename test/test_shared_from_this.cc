//
// Created by Qitong Wang on 2022/10/17.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "test_shared_from_this.h"

#include <random>

#include <boost/format.hpp>

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;
namespace test = upcite::dstree::test;

test::TestNode::TestNode(const std::shared_ptr<dstree::Config> &config,
                   const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                   ID_TYPE depth,
                   ID_TYPE id,
                   std::shared_ptr<test::TestNode> parent,
                   const std::shared_ptr<EAPCAEnvelope> &eapca_envelope) :
//    config_(config),
    depth_(depth),
    id_(id),
//    parent_(std::move(parent)),
    nseries_(0) {
//  buffer_ = std::move(buffer_manager->create_node_buffer(id_));

  is_leaf_ = true;

//  split_ = std::make_shared<dstree::Split>();
//  children_.reserve(config_->node_nchild_);

  if (eapca_envelope == nullptr) {
    assert(parent == nullptr); // only root is allowed to initialize a default EAPCA envelope

//    eapca_envelope_ = std::make_shared<EAPCAEnvelope>(config_, 1);
  } else {
//    eapca_envelope_ = std::make_shared<EAPCAEnvelope>(eapca_envelope);
  }
}

std::shared_ptr<test::TestNode> test::TestNode::route(const std::shared_ptr<dstree::EAPCA> &series_eapca,
                                                  const std::shared_ptr<upcite::Logger> &logger) const {
  ID_TYPE target_child_id = 0;

//  if (split_->is_vertical_split_) {
//    target_child_id = split_->route(series_eapca->get_subsegment_value(
//        split_->split_subsegment_id_, split_->horizontal_split_mode_ == MEAN));
//
//    series_eapca->split(config_, split_, eapca_envelope_->segment_lengths_, eapca_envelope_->subsegment_lengths_);
//  } else {
//    target_child_id = split_->route(series_eapca->get_segment_value(
//        split_->split_segment_id_, split_->horizontal_split_mode_ == MEAN));
//  }

  std::random_device rd;
  std::mt19937::result_type seed = rd() ^ (
      (std::mt19937::result_type)
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now().time_since_epoch()
          ).count() +
          (std::mt19937::result_type)
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now().time_since_epoch()
              ).count() );

  std::mt19937 gen(seed);
//  std::uniform_int_distribution<unsigned> distrib(0, config_->node_nchild_);
  std::uniform_int_distribution<unsigned> distrib(0, 1);

//  return children_[target_child_id];
//  return children_[distrib(gen)];
  return std::make_shared<TestNode>();
}

RESPONSE test::TestNode::insert(ID_TYPE series_id,
                              const std::shared_ptr<dstree::EAPCA> &series_eapca,
                              const std::shared_ptr<upcite::Logger> &logger) {
  // TODO optimize RESPONSE operators
//  RESPONSE response = buffer_->insert(series_id);
//
////  response = static_cast<RESPONSE>(response || eapca_envelope_->update(series_eapca));
//
//  if (response == SUCCESS) {
//    nseries_ += 1;
//  }
  nseries_ += 1;

  return FAILURE;
//  return response;
}

RESPONSE test::TestNode::split(const std::shared_ptr<dstree::Config> &config,
                             const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                             ID_TYPE first_child_id,
                             const std::shared_ptr<upcite::Logger> &logger) {
  std::shared_ptr<dstree::Split> current_split = std::make_shared<dstree::Split>();
  VALUE_TYPE mean_width, max_std, min_mean, min_std;
  VALUE_TYPE mean_width_children, std_width_children, max_std_child;
  VALUE_TYPE range_parent, range_children;
  VALUE_TYPE quality_gain, best_so_far_quality_gain = constant::MIN_VALUE, best_so_far_quality_gain_vertical;
//  VALUE_TYPE segment_length, subsegment_length, nchild = static_cast<VALUE_TYPE>(config_->node_nchild_);
  VALUE_TYPE segment_length, subsegment_length, nchild = static_cast<VALUE_TYPE>(2);

  ID_TYPE segment_id, subsegment_id, child_id;
//  for (segment_id = 0; segment_id < eapca_envelope_->nsegment_; ++segment_id) {
//    current_split->split_segment_id_ = segment_id;
//
//    mean_width = eapca_envelope_->segment_max_means_[segment_id] - eapca_envelope_->segment_min_means_[segment_id];
//    max_std = eapca_envelope_->segment_max_stds_[segment_id];
//    segment_length = static_cast<VALUE_TYPE>(eapca_envelope_->segment_lengths_[segment_id]);
//
//    range_parent = segment_length * (mean_width * mean_width + max_std * max_std);
//
//    current_split->is_vertical_split_ = false;
//
//    current_split->horizontal_split_mode_ = MEAN;
//    mean_width_children = mean_width / nchild;
//    range_children = 0;
//
//    for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
//      range_children += segment_length * (mean_width_children * mean_width_children + max_std * max_std);
//    }
//
//    quality_gain = range_parent - range_children / nchild;
//
//    if (quality_gain > best_so_far_quality_gain) {
//      best_so_far_quality_gain = quality_gain;
//      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;
//
//      min_mean = eapca_envelope_->segment_min_means_[segment_id];
//      current_split->horizontal_breakpoints_.clear();
//      for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
//        current_split->horizontal_breakpoints_.push_back(
//            min_mean + mean_width_children * static_cast<VALUE_TYPE>(child_id));
//      }
//
//      *split_ = *current_split;
//    }
//
//    current_split->horizontal_split_mode_ = STD;
//    std_width_children = (max_std - eapca_envelope_->segment_min_stds_[segment_id]) / nchild;
//    range_children = 0;
//
//    for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
//      max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
//      range_children += segment_length * (mean_width * mean_width + max_std_child * max_std_child);
//    }
//
//    quality_gain = range_parent - range_children / nchild;
//
//    if (quality_gain > best_so_far_quality_gain) {
//      best_so_far_quality_gain = quality_gain;
//      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;
//
//      min_std = eapca_envelope_->segment_min_stds_[segment_id];
//      current_split->horizontal_breakpoints_.clear();
//      for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
//        current_split->horizontal_breakpoints_.push_back(
//            min_std + std_width_children * static_cast<VALUE_TYPE>(child_id));
//      }
//
//      *split_ = *current_split;
//    }
//
//    current_split->is_vertical_split_ = true;
//
//    for (subsegment_id = segment_id * config_->vertical_split_nsubsegment_;
//         subsegment_id < (segment_id + 1) * config_->vertical_split_nsubsegment_;
//         ++subsegment_id) {
//      current_split->split_subsegment_id_ = subsegment_id;
//
//      mean_width =
//          eapca_envelope_->subsegment_max_means_[subsegment_id] - eapca_envelope_->subsegment_min_means_[subsegment_id];
//      max_std = eapca_envelope_->subsegment_max_stds_[subsegment_id];
//      subsegment_length = static_cast<VALUE_TYPE>(eapca_envelope_->subsegment_lengths_[subsegment_id]);
//
//      range_parent = subsegment_length * (mean_width * mean_width + max_std * max_std);
//
//      current_split->horizontal_split_mode_ = MEAN;
//      mean_width_children = mean_width / nchild;
//      range_children = 0;
//
//      for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
//        range_children += subsegment_length * (mean_width_children * mean_width_children + max_std * max_std);
//      }
//
//      quality_gain = range_parent - range_children / nchild;
//
//      if (quality_gain > best_so_far_quality_gain_vertical) {
//        best_so_far_quality_gain_vertical = quality_gain;
//        best_so_far_quality_gain = best_so_far_quality_gain_vertical / config_->vertical_split_gain_tradeoff_factor_;
//
//        min_mean = eapca_envelope_->subsegment_min_means_[subsegment_id];
//        current_split->horizontal_breakpoints_.clear();
//        for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
//          current_split->horizontal_breakpoints_.push_back(
//              min_mean + mean_width_children * static_cast<VALUE_TYPE>(child_id));
//        }
//
//        *split_ = *current_split;
//      }
//
//      current_split->horizontal_split_mode_ = STD;
//      std_width_children = (max_std - eapca_envelope_->subsegment_min_stds_[subsegment_id]) / nchild;
//      range_children = 0;
//
//      for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
//        max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
//        range_children += subsegment_length * (mean_width * mean_width + max_std_child * max_std_child);
//      }
//
//      quality_gain = range_parent - range_children / nchild;
//
//      if (quality_gain > best_so_far_quality_gain_vertical) {
//        best_so_far_quality_gain_vertical = quality_gain;
//        best_so_far_quality_gain = best_so_far_quality_gain_vertical / config_->vertical_split_gain_tradeoff_factor_;
//
//        min_std = eapca_envelope_->subsegment_min_stds_[subsegment_id];
//        current_split->horizontal_breakpoints_.clear();
//        for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
//          current_split->horizontal_breakpoints_.push_back(
//              min_std + std_width_children * static_cast<VALUE_TYPE>(child_id));
//        }
//
//        *split_ = *current_split;
//      }
//    }
//  }

//  split_->split_segment_offset_ = 0;
//  if (split_->is_vertical_split_) {
//    for (subsegment_id = 0; subsegment_id < split_->split_subsegment_id_; ++subsegment_id) {
//      split_->split_segment_offset_ += eapca_envelope_->subsegment_lengths_[subsegment_id];
//    }
//    split_->split_segment_length_ = eapca_envelope_->subsegment_lengths_[subsegment_id];
//  } else {
//    for (segment_id = 0; segment_id < split_->split_segment_id_; ++segment_id) {
//      split_->split_segment_offset_ += eapca_envelope_->segment_lengths_[segment_id];
//    }
//    split_->split_segment_length_ = eapca_envelope_->segment_lengths_[segment_id];
//  }

//  std::weak_ptr<dstree::TestNode> parent = shared_from_this();
//  auto parent = shared_from_this();
//  std::shared_ptr<dstree::EAPCAEnvelope> child_eapca_envelope = std::make_shared<dstree::EAPCAEnvelope>(
//      config, eapca_envelope_, split_, logger);

//  for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
//    children_.push_back(std::make_shared<test::TestNode>(
//        config, buffer_manager, depth_ + 1, first_child_id + child_id, parent));
////        config, buffer_manager, depth_ + 1, first_child_id + child_id, parent, child_eapca_envelope));
//  }

//  for (ID_TYPE node_series_id = 0; node_series_id < buffer_->size(); ++node_series_id) {
//    ID_TYPE series_batch_id = buffer_->get_offset(node_series_id);
//
//    std::shared_ptr<test::TestNode> target_child = route(buffer_manager->batch_eapca_[series_batch_id], logger);
//    target_child->insert(series_batch_id, buffer_manager->batch_eapca_[series_batch_id], logger);
//  }

  // TODO load flushed series, calculate their EAPCA and insert into children TestNodes

//  buffer_->clean(true);
  nseries_ = 0;

  is_leaf_ = false;

  return SUCCESS;
}

int main(int argc, char *argv[]) {
  std::shared_ptr<dstree::Config> config =  std::make_shared<dstree::Config>(argc, argv);
  std::shared_ptr<upcite::Logger> logger = std::make_shared<upcite::Logger>(config->log_filepath_);
  config->log(logger);

  auto buffer_manager = std::make_unique<dstree::BufferManager>(config, logger);
  ID_TYPE node_id_accumulator = 0;

  auto root = std::make_shared<test::TestNode>(config, buffer_manager, 0, node_id_accumulator);
  node_id_accumulator += 1;

//  auto parent = root->shared_from_this();

//  while (buffer_manager->load_batch() == SUCCESS) {
//    for (ID_TYPE series_id = 0; series_id < buffer_manager->load_buffer_size(); ++series_id) {
//      buffer_manager->batch_eapca_.push_back(std::make_shared<dstree::EAPCA>(
//          buffer_manager->get_series_ptr(series_id),
//          config->series_length_,
//          config->vertical_split_nsubsegment_));
//      std::shared_ptr<dstree::EAPCA> series_eapca = buffer_manager->batch_eapca_[series_id];
//
//      std::shared_ptr<test::TestNode> target_node = root;
//
//      while (!target_node->is_leaf()) {
//        target_node = target_node->route(series_eapca, logger);
//      }
//
//      if (target_node->is_full()) {
//        target_node->split(config, buffer_manager, node_id_accumulator, logger);
//        node_id_accumulator += config->node_nchild_;
//
//        target_node = target_node->route(series_eapca, logger);
//      }
//
//      target_node->insert(series_id, series_eapca, logger);
//    }
//
//    if (config->on_disk_) {
//      buffer_manager->flush();
//    }
//  }

//  if (!buffer_manager->is_fully_loaded()) {
//    return FAILURE;
//  }

  return SUCCESS;
}
