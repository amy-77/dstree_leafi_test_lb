//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "node.h"

#include <utility>
#include <iostream>

#include <boost/format.hpp>

#include "stat.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Node::Node(std::shared_ptr<dstree::Config> config,
                   const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                   ID_TYPE depth,
                   ID_TYPE id,
                   const std::shared_ptr<EAPCAEnvelope> &eapca_envelope) :
    config_(std::move(config)),
    depth_(depth),
    id_(id),
    nseries_(0) {
  buffer_ = buffer_manager->create_node_buffer(id_);

  split_ = std::make_shared<dstree::Split>();
  children_.reserve(config_->node_nchild_);

  if (eapca_envelope == nullptr) {
//    assert(parent == nullptr); // only root is allowed to initialize a default EAPCA envelope

    eapca_envelope_ = std::make_shared<EAPCAEnvelope>(config_, 1);
  } else {
    eapca_envelope_ = std::make_shared<EAPCAEnvelope>(eapca_envelope);
  }
}

std::shared_ptr<dstree::Node> dstree::Node::route(const std::shared_ptr<dstree::EAPCA> &series_eapca,
                                                  const std::shared_ptr<upcite::Logger> &logger) const {
  ID_TYPE target_child_id = 0;

#ifdef DEBUG
#ifndef DEBUGGED
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("before: vertical_split = %d, node %d - %d == series %d - %d")
          % split_->is_vertical_split_
          % eapca_envelope_->nsegment_
          % eapca_envelope_->nsubsegment_
          % series_eapca->nsegment_
          % series_eapca->nsubsegment_;
  }
#endif
#endif

  if (split_->is_vertical_split_) {
    target_child_id = split_->route(series_eapca->get_subsegment_value(
        split_->split_subsegment_id_, split_->horizontal_split_mode_ == MEAN));

    series_eapca->split(config_, split_, eapca_envelope_->segment_lengths_, eapca_envelope_->subsegment_lengths_);
  } else {
    target_child_id = split_->route(series_eapca->get_segment_value(
        split_->split_segment_id_, split_->horizontal_split_mode_ == MEAN));
  }

#ifdef DEBUG
#ifndef DEBUGGED
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("after: vertical_split = %d, node %d - %d == series %d - %d")
          % split_->is_vertical_split_
          % eapca_envelope_->nsegment_
          % eapca_envelope_->nsubsegment_
          % series_eapca->nsegment_
          % series_eapca->nsubsegment_;
  }
#endif
#endif

  return children_[target_child_id];
}

std::shared_ptr<dstree::Node> dstree::Node::route(const VALUE_TYPE *series_ptr) const {
  ID_TYPE target_child_id = 0;

  VALUE_TYPE mean, std;
  std::tie(mean, std) = upcite::cal_mean_std(
      series_ptr + split_->split_segment_offset_, split_->split_segment_length_);
  target_child_id = split_->route(split_->horizontal_split_mode_ == MEAN ? mean : std);

  return children_[target_child_id];
}

RESPONSE dstree::Node::enqueue_leaf(std::vector<std::shared_ptr<Node>> &leaves) {
  if (is_leaf()) {
    leaves.push_back(shared_from_this());
  } else {
    for (auto child_node : children_) {
      child_node->enqueue_leaf(leaves);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Node::insert(ID_TYPE series_id,
                              const std::shared_ptr<dstree::EAPCA> &series_eapca,
                              const std::shared_ptr<upcite::Logger> &logger) {
  // TODO optimize RESPONSE operators
  RESPONSE response = buffer_->insert(series_id, logger);

#ifdef DEBUG
#ifndef DEBUGGED
  if (response == FAILURE && logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("node %d: %d - %d == series %d: %d - %d")
          % id_
          % eapca_envelope_->nsegment_
          % eapca_envelope_->nsubsegment_
          % series_id
          % series_eapca->nsegment_
          % series_eapca->nsubsegment_;
  }
#endif
#endif
  response = static_cast<RESPONSE>(response || eapca_envelope_->update(series_eapca));

  if (response == SUCCESS) {
    nseries_ += 1;
  } else {
#ifdef DEBUG
#ifndef DEBUGGED
    if (logger != nullptr) {
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format("node %d: %d - %d == series %d: %d - %d")
            % id_
            % eapca_envelope_->nsegment_
            % eapca_envelope_->nsubsegment_
            % series_id
            % series_eapca->nsegment_
            % series_eapca->nsubsegment_;
    }
#endif
#endif
  }

#ifdef DEBUG
#ifndef DEBUGGED
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("id_ = %d, nseries_ = %d, buffer_->size() = %d")
          % id_
          % nseries_
          % buffer_->size();
  }

  assert(nseries_ == buffer_->size());
#endif
#endif

  return response;
}

RESPONSE dstree::Node::split(const std::shared_ptr<dstree::Config> &config,
                             const std::unique_ptr<dstree::BufferManager> &buffer_manager,
                             ID_TYPE first_child_id,
                             const std::shared_ptr<upcite::Logger> &logger) {
  std::shared_ptr<dstree::Split> current_split = std::make_shared<dstree::Split>();
  VALUE_TYPE mean_width, max_std, min_mean, min_std;
  VALUE_TYPE mean_width_children, std_width_children, max_std_child;
  VALUE_TYPE range_parent, range_children;
  VALUE_TYPE quality_gain, best_so_far_quality_gain = constant::MIN_VALUE, best_so_far_quality_gain_vertical;
  VALUE_TYPE segment_length, subsegment_length, nchild = static_cast<VALUE_TYPE>(config_->node_nchild_);

  ID_TYPE segment_id, subsegment_id, child_id;
  for (segment_id = 0; segment_id < eapca_envelope_->nsegment_; ++segment_id) {
    current_split->split_segment_id_ = segment_id;

    mean_width = eapca_envelope_->segment_max_means_[segment_id] - eapca_envelope_->segment_min_means_[segment_id];
    max_std = eapca_envelope_->segment_max_stds_[segment_id];
    segment_length = static_cast<VALUE_TYPE>(eapca_envelope_->segment_lengths_[segment_id]);

    range_parent = segment_length * (mean_width * mean_width + max_std * max_std);

    current_split->is_vertical_split_ = false;

    current_split->horizontal_split_mode_ = MEAN;
    mean_width_children = mean_width / nchild;
    range_children = 0;

    for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
      range_children += segment_length * (mean_width_children * mean_width_children + max_std * max_std);
    }

    quality_gain = range_parent - range_children / nchild;

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format(
          "node_id = %d, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
          % id_
          % current_split->is_vertical_split_
          % current_split->split_segment_id_
          % current_split->split_subsegment_id_
          % current_split->horizontal_split_mode_;
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format(
          "quality_gain = %f, range_parent = %f, range_children / nchild = %f")
          % quality_gain
          % range_parent
          % (range_children / nchild);
#endif
#endif

    if (quality_gain > best_so_far_quality_gain) {
      best_so_far_quality_gain = quality_gain;
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;

      min_mean = eapca_envelope_->segment_min_means_[segment_id];
      current_split->horizontal_breakpoints_.clear();
      for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
        current_split->horizontal_breakpoints_.push_back(
            min_mean + mean_width_children * static_cast<VALUE_TYPE>(child_id));
      }

      *split_ = *current_split;

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "node_id = %d, best_so_far_quality_gain = %f, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
            % id_
            % best_so_far_quality_gain
            % current_split->is_vertical_split_
            % current_split->split_segment_id_
            % current_split->split_subsegment_id_
            % current_split->horizontal_split_mode_;
#endif
#endif
    }

    current_split->horizontal_split_mode_ = STD;
    std_width_children = (max_std - eapca_envelope_->segment_min_stds_[segment_id]) / nchild;
    range_children = 0;

    for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
      max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
      range_children += segment_length * (mean_width * mean_width + max_std_child * max_std_child);
    }

    quality_gain = range_parent - range_children / nchild;

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format(
          "node_id = %d, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
          % id_
          % current_split->is_vertical_split_
          % current_split->split_segment_id_
          % current_split->split_subsegment_id_
          % current_split->horizontal_split_mode_;
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format(
          "quality_gain = %f, range_parent = %f, range_children / nchild = %f")
          % quality_gain
          % range_parent
          % (range_children / nchild);
#endif
#endif

    if (quality_gain > best_so_far_quality_gain) {
      best_so_far_quality_gain = quality_gain;
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;

      min_std = eapca_envelope_->segment_min_stds_[segment_id];
      current_split->horizontal_breakpoints_.clear();
      for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
        current_split->horizontal_breakpoints_.push_back(
            min_std + std_width_children * static_cast<VALUE_TYPE>(child_id));
      }

      *split_ = *current_split;

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "node_id = %d, best_so_far_quality_gain = %f, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
            % id_
            % best_so_far_quality_gain
            % current_split->is_vertical_split_
            % current_split->split_segment_id_
            % current_split->split_subsegment_id_
            % current_split->horizontal_split_mode_;
#endif
#endif
    }

    current_split->is_vertical_split_ = true;

    for (subsegment_id = segment_id * config_->vertical_split_nsubsegment_;
         subsegment_id < (segment_id + 1) * config_->vertical_split_nsubsegment_;
         ++subsegment_id) {
      current_split->split_subsegment_id_ = subsegment_id;

      mean_width =
          eapca_envelope_->subsegment_max_means_[subsegment_id] - eapca_envelope_->subsegment_min_means_[subsegment_id];
      max_std = eapca_envelope_->subsegment_max_stds_[subsegment_id];
      subsegment_length = static_cast<VALUE_TYPE>(eapca_envelope_->subsegment_lengths_[subsegment_id]);

      range_parent = subsegment_length * (mean_width * mean_width + max_std * max_std);

      current_split->horizontal_split_mode_ = MEAN;
      mean_width_children = mean_width / nchild;
      range_children = 0;

      for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
        range_children += subsegment_length * (mean_width_children * mean_width_children + max_std * max_std);
      }

      quality_gain = range_parent - range_children / nchild;

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "node_id = %d, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
            % id_
            % current_split->is_vertical_split_
            % current_split->split_segment_id_
            % current_split->split_subsegment_id_
            % current_split->horizontal_split_mode_;
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "quality_gain = %f, range_parent = %f, range_children / nchild = %f")
            % quality_gain
            % range_parent
            % (range_children / nchild);
#endif
#endif

      if (quality_gain > best_so_far_quality_gain_vertical) {
        best_so_far_quality_gain_vertical = quality_gain;
        best_so_far_quality_gain = best_so_far_quality_gain_vertical / config_->vertical_split_gain_tradeoff_factor_;

        min_mean = eapca_envelope_->subsegment_min_means_[subsegment_id];
        current_split->horizontal_breakpoints_.clear();
        for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
          current_split->horizontal_breakpoints_.push_back(
              min_mean + mean_width_children * static_cast<VALUE_TYPE>(child_id));
        }

        *split_ = *current_split;

#ifdef DEBUG
#ifndef DEBUGGED
        MALAT_LOG(logger->logger, trivial::debug)
          << boost::format(
              "node_id = %d, best_so_far_quality_gain = %f, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
              % id_
              % best_so_far_quality_gain
              % current_split->is_vertical_split_
              % current_split->split_segment_id_
              % current_split->split_subsegment_id_
              % current_split->horizontal_split_mode_;
#endif
#endif
      }

      current_split->horizontal_split_mode_ = STD;
      std_width_children = (max_std - eapca_envelope_->subsegment_min_stds_[subsegment_id]) / nchild;
      range_children = 0;

      for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
        max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
        range_children += subsegment_length * (mean_width * mean_width + max_std_child * max_std_child);
      }

      quality_gain = range_parent - range_children / nchild;

#ifdef DEBUG
#ifndef DEBUGGED
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "node_id = %d, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
            % id_
            % current_split->is_vertical_split_
            % current_split->split_segment_id_
            % current_split->split_subsegment_id_
            % current_split->horizontal_split_mode_;
      MALAT_LOG(logger->logger, trivial::debug)
        << boost::format(
            "quality_gain = %f, range_parent = %f, range_children / nchild = %f")
            % quality_gain
            % range_parent
            % (range_children / nchild);
#endif
#endif

      if (quality_gain > best_so_far_quality_gain_vertical) {
        best_so_far_quality_gain_vertical = quality_gain;
        best_so_far_quality_gain = best_so_far_quality_gain_vertical / config_->vertical_split_gain_tradeoff_factor_;

        min_std = eapca_envelope_->subsegment_min_stds_[subsegment_id];
        current_split->horizontal_breakpoints_.clear();
        for (child_id = 1; child_id < config_->node_nchild_; ++child_id) {
          current_split->horizontal_breakpoints_.push_back(
              min_std + std_width_children * static_cast<VALUE_TYPE>(child_id));
        }

        *split_ = *current_split;

#ifdef DEBUG
#ifndef DEBUGGED
        MALAT_LOG(logger->logger, trivial::debug)
          << boost::format(
              "node_id = %d, best_so_far_quality_gain = %f, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, horizontal_split_mode_ = %d")
              % id_
              % best_so_far_quality_gain
              % current_split->is_vertical_split_
              % current_split->split_segment_id_
              % current_split->split_subsegment_id_
              % current_split->horizontal_split_mode_;
#endif
#endif
      }
    }
  }

  split_->split_segment_offset_ = 0;
  if (split_->is_vertical_split_) {
    for (subsegment_id = 0; subsegment_id < split_->split_subsegment_id_; ++subsegment_id) {
      split_->split_segment_offset_ += eapca_envelope_->subsegment_lengths_[subsegment_id];
    }
    split_->split_segment_length_ = eapca_envelope_->subsegment_lengths_[subsegment_id];
  } else {
    for (segment_id = 0; segment_id < split_->split_segment_id_; ++segment_id) {
      split_->split_segment_offset_ += eapca_envelope_->segment_lengths_[segment_id];
    }
    split_->split_segment_length_ = eapca_envelope_->segment_lengths_[segment_id];
  }

  // error for a local-scope shared_ptr, which is on stack instead of heap
//  std::shared_ptr<dstree::Node> parent(this);
  // TODO libc++abi: terminating with uncaught exception of type std::__1::bad_weak_ptr: bad_weak_ptr
//  std::shared_ptr<dstree::Node> parent = shared_from_this();

  std::shared_ptr<dstree::EAPCAEnvelope> child_eapca_envelope = std::make_shared<dstree::EAPCAEnvelope>(
      config, eapca_envelope_, split_, logger);

  for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
    children_.push_back(std::make_shared<dstree::Node>(
        config, buffer_manager, depth_ + 1, first_child_id + child_id, child_eapca_envelope));

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("child_id = %d, %d - %d -> %d - %d")
          % (first_child_id + child_id)
          % eapca_envelope_->nsegment_
          % eapca_envelope_->nsubsegment_
          % children_[child_id]->eapca_envelope_->nsegment_
          % children_[child_id]->eapca_envelope_->nsubsegment_;
#endif
#endif
  }

#ifdef DEBUG
#ifndef DEBUGGED
  for (ID_TYPE node_series_id = 0; node_series_id < buffer_->size(); node_series_id += 20) {
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("%d(%d): %d %d %d %d %d - %d %d %d %d %d - %d %d %d %d %d - %d %d %d %d %d")
          % node_series_id
          % id_
          % buffer_->get_offset(node_series_id)
          % buffer_->get_offset(node_series_id + 1)
          % buffer_->get_offset(node_series_id + 2)
          % buffer_->get_offset(node_series_id + 3)
          % buffer_->get_offset(node_series_id + 4)
          % buffer_->get_offset(node_series_id + 5)
          % buffer_->get_offset(node_series_id + 6)
          % buffer_->get_offset(node_series_id + 7)
          % buffer_->get_offset(node_series_id + 8)
          % buffer_->get_offset(node_series_id + 9)
          % buffer_->get_offset(node_series_id + 10)
          % buffer_->get_offset(node_series_id + 11)
          % buffer_->get_offset(node_series_id + 12)
          % buffer_->get_offset(node_series_id + 13)
          % buffer_->get_offset(node_series_id + 14)
          % buffer_->get_offset(node_series_id + 15)
          % buffer_->get_offset(node_series_id + 16)
          % buffer_->get_offset(node_series_id + 17)
          % buffer_->get_offset(node_series_id + 18)
          % buffer_->get_offset(node_series_id + 19);
  }
#endif
#endif

  for (ID_TYPE node_series_id = 0; node_series_id < buffer_->size(); ++node_series_id) {
    ID_TYPE series_batch_id = buffer_->get_offset(node_series_id);

//#ifdef DEBUG
//#ifndef DEBUGGED
//    MALAT_LOG(logger->logger, trivial::debug)
//      << boost::format("node %d(%d): %d - %d == series %d/%d: %d - %d")
//          % id_
//          % split_->is_vertical_split_
//          % eapca_envelope_->nsegment_
//          % eapca_envelope_->nsubsegment_
//          % series_batch_id
//          % node_series_id
//          % buffer_manager->batch_eapca_[series_batch_id]->nsegment_
//          % buffer_manager->batch_eapca_[series_batch_id]->nsubsegment_;
//#endif
//#endif

    std::shared_ptr<dstree::Node> target_child = route(buffer_manager->batch_eapca_[series_batch_id], logger);
    target_child->insert(series_batch_id, buffer_manager->batch_eapca_[series_batch_id], logger);

//#ifdef DEBUG
//#ifndef DEBUGGED
//    MALAT_LOG(logger->logger, trivial::debug) << boost::format(
//          "node %d(%d): %d - %d -> %d: %d - %d == series %d/%d: %d - %d")
//          % id_
//          % split_->is_vertical_split_
//          % eapca_envelope_->nsegment_
//          % eapca_envelope_->nsubsegment_
//          % target_child->id_
//          % target_child->eapca_envelope_->nsegment_
//          % target_child->eapca_envelope_->nsubsegment_
//          % series_batch_id
//          % node_series_id
//          % buffer_manager->batch_eapca_[series_batch_id]->nsegment_
//          % buffer_manager->batch_eapca_[series_batch_id]->nsubsegment_;
//#endif
//#endif
  }

#ifdef DEBUG
//#ifndef DEBUGGED
  MALAT_LOG(logger->logger, trivial::debug) << boost::format(
        "parent %d, split %d-%d @ %.3f = %.3f: (%d in) %d @ %d + %d, child %d (%d == %d) + %d (%d == %d)")
        % id_
        % split_->is_vertical_split_
        % split_->horizontal_split_mode_
        % split_->horizontal_breakpoints_[0]
        % best_so_far_quality_gain
        % split_->split_subsegment_id_
        % split_->split_segment_id_
        % split_->split_segment_offset_
        % split_->split_segment_length_
        % children_[0]->id_
        % children_[0]->nseries_
        % children_[0]->buffer_->size()
        % children_[1]->id_
        % children_[1]->nseries_
        % children_[1]->buffer_->size();
//#endif
#endif

  // TODO load flushed series, calculate their EAPCA and insert into children nodes

  buffer_->clean(true);
  nseries_ = 0;

  return SUCCESS;
}

RESPONSE dstree::Node::search(const VALUE_TYPE *series_ptr,
                              std::shared_ptr<dstree::Answer> &answer,
                              ID_TYPE resident_node_id) const {
  if (resident_node_id != id_) {
    ;
  }

  return SUCCESS;
}

RESPONSE dstree::Node::log(const std::shared_ptr<upcite::Logger> &logger) {
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("node_id = %d, node_depth = %d, node_size = %d") % id_ % depth_ % nseries_;

  return SUCCESS;
}