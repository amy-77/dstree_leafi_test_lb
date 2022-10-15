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
                   std::shared_ptr<dstree::Node> parent,
                   const std::shared_ptr<EAPCA_Envelope> &eapca_envelope) :
    config_(std::move(config)),
    parent_(std::move(parent)),
    depth_(depth),
    id_(id),
    nseries_(0) {
  buffer_ = buffer_manager->create_node_buffer(id_);

  split_ = std::make_shared<dstree::Split>();
  children_.reserve(config_->node_nchild_);

  if (eapca_envelope == nullptr) {
    assert(parent == nullptr); // only root is allowed to initialize a default EAPCA envelope

    eapca_envelope_ = std::make_shared<EAPCA_Envelope>(config_, 1);
  } else {
    eapca_envelope_ = std::make_shared<EAPCA_Envelope>(eapca_envelope);
  }
}

std::shared_ptr<dstree::Node> dstree::Node::route(const std::shared_ptr<dstree::EAPCA> &series_eapca) const {
  ID_TYPE target_child_id = 0;

  if (split_->is_vertical_split_) {
    series_eapca->split(config_, split_, eapca_envelope_->segment_lengths_, eapca_envelope_->subsegment_lengths_);

    target_child_id = split_->route(series_eapca->get_segment_value(
        split_->split_subsegment_id_, split_->horizontal_split_mode_ == MEAN));
  } else {
    target_child_id = split_->route(series_eapca->get_segment_value(
        split_->split_segment_id_, split_->horizontal_split_mode_ == MEAN));
  }

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

RESPONSE dstree::Node::insert(ID_TYPE series_id, const std::shared_ptr<dstree::EAPCA> &series_eapca) {
  // TODO optimize RESPONSE operators
  RESPONSE response = buffer_->insert(series_id);
  response = static_cast<RESPONSE>(response || eapca_envelope_->update(series_eapca));

  if (response == SUCCESS) {
    nseries_ += 1;
  }

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

    mean_width = eapca_envelope_->max_means_[segment_id] - eapca_envelope_->min_means_[segment_id];
    max_std = eapca_envelope_->max_stds_[segment_id];
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
    //#ifndef DEBUGGED
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
//#endif
#endif

    if (quality_gain > best_so_far_quality_gain) {
      best_so_far_quality_gain = quality_gain;
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;

      min_mean = eapca_envelope_->min_means_[segment_id];
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
    std_width_children = (max_std - eapca_envelope_->min_stds_[segment_id]) / nchild;
    range_children = 0;

    for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
      max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
      range_children += segment_length * (mean_width * mean_width + max_std_child * max_std_child);
    }

    quality_gain = range_parent - range_children / nchild;

#ifdef DEBUG
//#ifndef DEBUGGED
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
//#endif
#endif

    if (quality_gain > best_so_far_quality_gain) {
      best_so_far_quality_gain = quality_gain;
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_->vertical_split_gain_tradeoff_factor_;

      min_std = eapca_envelope_->min_stds_[segment_id];
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
      //#ifndef DEBUGGED
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
//#endif
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
      //#ifndef DEBUGGED
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
//#endif
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

#ifdef DEBUG
//#ifndef DEBUGGED
  MALAT_LOG(logger->logger, trivial::debug)
    << boost::format(
        "node_id = %d, best_so_far_quality_gain = %f, is_vertical_split_ = %d, split_segment_id_ = %d, split_subsegment_id_ = %d, split_segment_offset_ = %d, split_segment_length_ = %d, horizontal_split_mode_ = %d, horizontal_breakpoints_ = %d")
        % id_
        % best_so_far_quality_gain
        % split_->is_vertical_split_
        % split_->split_segment_id_
        % split_->split_subsegment_id_
        % split_->split_segment_offset_
        % split_->split_segment_length_
        % split_->horizontal_split_mode_
        % split_->horizontal_breakpoints_[0];
//#endif
#endif

  std::shared_ptr<dstree::Node> parent(this);
  std::shared_ptr<dstree::EAPCA_Envelope> child_eapca_envelope = std::make_shared<dstree::EAPCA_Envelope>(
      config, eapca_envelope_, split_, logger);

  for (child_id = 0; child_id < config_->node_nchild_; ++child_id) {
    children_.push_back(std::make_shared<dstree::Node>(
        config, buffer_manager, depth_ + 1, first_child_id + child_id, parent, child_eapca_envelope));

#ifdef DEBUG
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("child_id = %d, nsegment = %d, nsubsegment = %d")
          % (first_child_id + child_id)
          % children_[child_id]->eapca_envelope_->nsegment_
          % children_[child_id]->eapca_envelope_->nsubsegment_;
#endif
  }

  for (ID_TYPE node_series_id = 0; node_series_id < buffer_->size(); ++node_series_id) {
    ID_TYPE series_batch_id = buffer_->offsets_[node_series_id];

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger->logger, trivial::debug)
      << boost::format("node_id = %d, series_id = %d, nsegment = %d, nsubsegment = %d")
          % id_
          % series_batch_id
          % buffer_manager->batch_eapca_[series_batch_id]->nsegment_
          % buffer_manager->batch_eapca_[series_batch_id]->nsubsegment_;
#endif
#endif

    std::shared_ptr<dstree::Node> target_child = route(buffer_manager->batch_eapca_[series_batch_id]);
    target_child->insert(series_batch_id, buffer_manager->batch_eapca_[series_batch_id]);
  }

  // TODO load flushed series, calculate their EAPCA and insert into children nodes

  buffer_->clean(true);
  nseries_ = 0;

  return SUCCESS;
}

RESPONSE dstree::Node::log(const std::shared_ptr<upcite::Logger> &logger) {
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("node_id = %d, node_depth = %d, node_size = %d") % id_ % depth_ % nseries_;

  return SUCCESS;
}
