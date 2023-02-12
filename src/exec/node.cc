//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "node.h"

#include <iostream>
#include <fstream>
#include <utility>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>

#include "stat.h"
#include "distance.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Node::Node(dstree::Config &config,
                   dstree::BufferManager &buffer_manager,
                   ID_TYPE depth,
                   ID_TYPE id) :
    config_(config),
    depth_(depth),
    id_(id),
    buffer_(buffer_manager.create_node_buffer(id)),
    nseries_(0),
    neurofilter_(nullptr) {
  ;
  split_ = std::make_unique<dstree::Split>();
  children_.reserve(config.node_nchild_);

  eapca_envelope_ = std::make_unique<EAPCAEnvelope>(config, 1);
}

dstree::Node::Node(dstree::Config &config,
                   dstree::BufferManager &buffer_manager,
                   ID_TYPE depth,
                   ID_TYPE id,
                   EAPCAEnvelope &eapca_envelope) :
    config_(config),
    depth_(depth),
    id_(id),
    buffer_(buffer_manager.create_node_buffer(id)),
    nseries_(0),
    neurofilter_(nullptr) {
  split_ = std::make_unique<dstree::Split>();
  children_.reserve(config.node_nchild_);

  eapca_envelope_ = std::make_unique<EAPCAEnvelope>(eapca_envelope);
}

dstree::Node &dstree::Node::route(dstree::EAPCA &series_eapca) const {
  ID_TYPE target_child_id;

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
    target_child_id = split_->route(series_eapca.get_subsegment_value(
        split_->split_subsegment_id_, split_->horizontal_split_mode_ == MEAN));

    series_eapca.split(config_, *split_, eapca_envelope_->segment_lengths_, eapca_envelope_->subsegment_lengths_);
  } else {
    target_child_id = split_->route(series_eapca.get_segment_value(
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

  return *children_[target_child_id];
}

dstree::Node &dstree::Node::route(const VALUE_TYPE *series_ptr) const {
  ID_TYPE target_child_id;

  if (split_->horizontal_split_mode_ == MEAN) {
    target_child_id = split_->route(upcite::cal_mean(
        series_ptr + split_->split_segment_offset_, split_->split_segment_length_));
  } else {
    target_child_id = split_->route(upcite::cal_std(
        series_ptr + split_->split_segment_offset_, split_->split_segment_length_));
  }

  return *children_[target_child_id];
}

RESPONSE dstree::Node::enqueue_leaf(std::vector<std::reference_wrapper<Node>> &leaves) {
  // TODO lshared_from_this() failed
  // ibc++abi: terminating with uncaught exception of type std::__1::bad_weak_ptr: bad_weak_ptr
//  if (is_leaf()) {
//    leaves.push_back(shared_from_this());
//  } else {
//    for (const auto& child_node : children_) {
//      child_node->enqueue_leaf(leaves);
//    }
//  }

  for (auto &child_node : children_) {
    if (child_node->is_leaf()) {
      leaves.push_back(std::ref(*child_node));
    } else {
      child_node->enqueue_leaf(leaves);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Node::insert(ID_TYPE series_id,
                              dstree::EAPCA &series_eapca) {
  // TODO optimize RESPONSE operators
  RESPONSE response = buffer_.get().insert(series_id);

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

RESPONSE dstree::Node::split(dstree::BufferManager &buffer_manager,
                             ID_TYPE first_child_id) {
  std::unique_ptr<dstree::Split> current_split = std::make_unique<dstree::Split>();
  VALUE_TYPE mean_width, max_std, min_mean, min_std;
  VALUE_TYPE mean_width_children, std_width_children, max_std_child;
  VALUE_TYPE range_parent, range_children;
  VALUE_TYPE quality_gain, best_so_far_quality_gain = constant::MIN_VALUE, best_so_far_quality_gain_vertical;
  VALUE_TYPE segment_length, subsegment_length, nchild = static_cast<VALUE_TYPE>(config_.get().node_nchild_);

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

    for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
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
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_.get().vertical_split_gain_tradeoff_factor_;

      min_mean = eapca_envelope_->segment_min_means_[segment_id];
      current_split->horizontal_breakpoints_.clear();
      for (child_id = 1; child_id < config_.get().node_nchild_; ++child_id) {
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

    for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
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
      best_so_far_quality_gain_vertical = best_so_far_quality_gain * config_.get().vertical_split_gain_tradeoff_factor_;

      min_std = eapca_envelope_->segment_min_stds_[segment_id];
      current_split->horizontal_breakpoints_.clear();
      for (child_id = 1; child_id < config_.get().node_nchild_; ++child_id) {
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

    for (subsegment_id = segment_id * config_.get().vertical_split_nsubsegment_;
         subsegment_id < (segment_id + 1) * config_.get().vertical_split_nsubsegment_;
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

      for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
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
        best_so_far_quality_gain =
            best_so_far_quality_gain_vertical / config_.get().vertical_split_gain_tradeoff_factor_;

        min_mean = eapca_envelope_->subsegment_min_means_[subsegment_id];
        current_split->horizontal_breakpoints_.clear();
        for (child_id = 1; child_id < config_.get().node_nchild_; ++child_id) {
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

      for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
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
        best_so_far_quality_gain =
            best_so_far_quality_gain_vertical / config_.get().vertical_split_gain_tradeoff_factor_;

        min_std = eapca_envelope_->subsegment_min_stds_[subsegment_id];
        current_split->horizontal_breakpoints_.clear();
        for (child_id = 1; child_id < config_.get().node_nchild_; ++child_id) {
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

  dstree::EAPCAEnvelope child_eapca_envelope(config_, *eapca_envelope_, *split_);

  for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
    children_.emplace_back(std::make_unique<dstree::Node>(config_, buffer_manager,
                                                          depth_ + 1, first_child_id + child_id,
                                                          child_eapca_envelope));

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

  for (ID_TYPE node_series_id = 0; node_series_id < buffer_.get().size(); ++node_series_id) {
    ID_TYPE series_batch_id = buffer_.get().get_offset(node_series_id);

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

    dstree::Node &target_child = route(buffer_manager.get_series_eapca(series_batch_id));
    target_child.insert(series_batch_id, buffer_manager.get_series_eapca(series_batch_id));

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
  spdlog::debug(
      "parent {:d}, split {:d}-{:d} @ {:.3f} = {:.3f}: ({:d} in) {:d} @ {:d} + {:d}, child {:d} ({:d} == {:d}) + {:d} ({:d} == {:d})",
      id_,
      split_->is_vertical_split_,
      static_cast<ID_TYPE>(split_->horizontal_split_mode_),
      split_->horizontal_breakpoints_[0],
      best_so_far_quality_gain,
      split_->split_subsegment_id_,
      split_->split_segment_id_,
      split_->split_segment_offset_,
      split_->split_segment_length_,
      children_[0]->id_,
      children_[0]->nseries_,
      children_[0]->buffer_.get().size(),
      children_[1]->id_,
      children_[1]->nseries_,
      children_[1]->buffer_.get().size());
//#endif
#endif

  // TODO load flushed series, calculate their EAPCA and insert into children nodes

  buffer_.get().clean(true);
  nseries_ = 0;

  return SUCCESS;
}

RESPONSE dstree::Node::search(const VALUE_TYPE *query_series_ptr,
                              dstree::Answer &answer,
                              ID_TYPE &visited_node_counter,
                              ID_TYPE &visited_series_counter) const {
  const VALUE_TYPE *db_series_ptr = buffer_.get().get_next_series_ptr();

  while (db_series_ptr != nullptr) {
    VALUE_TYPE distance = upcite::cal_EDsquare(db_series_ptr, query_series_ptr, config_.get().series_length_);

#ifdef DEBUG
#ifndef DEBUGGED
    MALAT_LOG(logger_->logger, trivial::debug) << boost::format(
          "query %d distance %.3f is_bsf %d bsf.size %d bsf.top %.3f")
          % answer->query_id_
          % distance
          % answer->is_bsf(distance)
          % answer->bsf_distances_.size()
          % answer->bsf_distances_.top();
#endif
#endif

    if (answer.is_bsf(distance)) {
      answer.push_bsf(distance);

      spdlog::info("query {:d} update bsf {:.3f} at node {:d} series {:d}",
                   answer.query_id_, distance, visited_node_counter, visited_series_counter);
    }

    visited_series_counter += 1;
    db_series_ptr = buffer_.get().get_next_series_ptr();
  }

  buffer_.get().reset();

  visited_node_counter += 1;

  return SUCCESS;
}

VALUE_TYPE dstree::Node::search(const VALUE_TYPE *query_series_ptr,
                                VALUE_TYPE *m256_fetch_cache,
                                VALUE_TYPE bsf_distance) const {
  const VALUE_TYPE *db_series_ptr = buffer_.get().get_next_series_ptr();
  VALUE_TYPE local_bsf = constant::MAX_VALUE;

  while (db_series_ptr != nullptr) {
    VALUE_TYPE distance = upcite::cal_EDsquare(db_series_ptr, query_series_ptr, config_.get().series_length_);

    // TODO resolve the conflict between multithreading and SIMD
//    if (bsf_distance > 0) {
//      distance = upcite::cal_early_EDsquare_SIMD_8(
//          db_series_ptr, query_series_ptr, config_.get().series_length_, m256_fetch_cache, bsf_distance);
//    } else {
//      distance = upcite::cal_EDsquare_SIMD_8(
//          db_series_ptr, query_series_ptr, config_.get().series_length_, m256_fetch_cache);
//    }

    if (distance < local_bsf) {
      local_bsf = distance;
    }

    db_series_ptr = buffer_.get().get_next_series_ptr();
  }

  buffer_.get().reset();

  return local_bsf;
}

RESPONSE dstree::Node::log() {
  spdlog::info("node {:d}: depth = {:d}, size = {:d}", id_, depth_, nseries_);

  for (ID_TYPE i = 0; i < children_.size(); ++i) {
    children_[i]->log();
  }

  return SUCCESS;
}

RESPONSE dstree::Node::dump(void *ofs_buf) const {
  std::string node_info_filepath = config_.get().dump_node_info_folderpath_ + std::to_string(id_) +
      config_.get().index_dump_file_postfix_;

  std::ofstream node_ofs(node_info_filepath, std::ios::out | std::ios::binary);
//  assert(node_fos.is_open());

  auto ofs_id_buf = reinterpret_cast<ID_TYPE *>(ofs_buf);

  node_ofs.write(reinterpret_cast<const char *>(&id_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(&depth_), sizeof(ID_TYPE));
  node_ofs.write(reinterpret_cast<const char *>(&nseries_), sizeof(ID_TYPE));

  eapca_envelope_->dump(node_ofs);

  ofs_id_buf[0] = static_cast<ID_TYPE>(children_.size());
  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));

  if (!children_.empty()) {
    for (ID_TYPE i = 0; i < children_.size(); ++i) {
      ofs_id_buf[i] = children_[i]->id_;
    }

    node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE) * children_.size());

    split_->dump(node_ofs, ofs_buf);
  }

  if (neurofilter_ != nullptr) {
    ofs_id_buf[0] = neurofilter_.get()->get_id();
  } else {
    ofs_id_buf[0] = -1;
  }
  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));

  if (neurofilter_ != nullptr) {
    neurofilter_->dump(node_ofs);
  }

  if (buffer_.get().size() > 0) {
    ofs_id_buf[0] = 1;
  } else {
    ofs_id_buf[0] = -1;
  }
  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));

  if (buffer_.get().size() > 0) {
    buffer_.get().dump();
  }

//  assert(node_fos.good());
  node_ofs.close();

  if (!children_.empty()) {
    for (ID_TYPE i = 0; i < children_.size(); ++i) {
      children_[i]->dump(ofs_buf);
    }
  }

  return SUCCESS;
}

RESPONSE dstree::Node::load(void *ifs_buf,
                            dstree::BufferManager &buffer_manager,
                            ID_TYPE &nnode,
                            ID_TYPE &nleaf) {
  std::string node_info_filepath = config_.get().load_node_info_folderpath_ + std::to_string(id_) +
      config_.get().index_dump_file_postfix_;

  if (!fs::is_regular_file(node_info_filepath)) {
    spdlog::error("Empty node_info_filepath found: {:s}", node_info_filepath);
    return FAILURE;
  }

  std::ifstream node_ifs(node_info_filepath, std::ios::in | std::ios::binary);
//  assert(node_ifs.is_open());

  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);

  // id_, depth_, nseries_
  ID_TYPE read_nbytes = sizeof(ID_TYPE) * 3;
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  id_ = ifs_id_buf[0];
  depth_ = ifs_id_buf[1];
  nseries_ = ifs_id_buf[2];

  RESPONSE status = eapca_envelope_->load(node_ifs, ifs_buf);

  if (status == FAILURE) {
    spdlog::error("node {:d} eapca_envelope loading failed", id_);
    node_ifs.close();
    return FAILURE;
  }

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE nchildren = ifs_id_buf[0];

  if (nchildren > 0) {
    read_nbytes = sizeof(ID_TYPE) * nchildren;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);

    for (ID_TYPE i = 0; i < nchildren; ++i) {
      children_.emplace_back(std::make_unique<dstree::Node>(config_, buffer_manager, depth_ + 1, ifs_id_buf[i]));
    }

    status = split_->load(node_ifs, ifs_buf);

    if (status == FAILURE) {
      spdlog::error("node {:d} split loading failed", id_);
      node_ifs.close();
      return FAILURE;
    }
  }

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE filter_id = ifs_id_buf[0];

  if (filter_id >= 0) {
    neurofilter_ = std::make_unique<dstree::Filter>(config_, filter_id, constant::TENSOR_PLACEHOLDER_REF);
    status = neurofilter_->load(node_ifs, ifs_buf);

    if (status == FAILURE) {
      spdlog::error("node {:d} neurofilter loading failed", id_);
      node_ifs.close();
      return FAILURE;
    }
  }

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE has_buffer = ifs_id_buf[0];

  if (has_buffer > 0) {
    status = buffer_.get().load(ifs_buf);

    if (status == FAILURE) {
      spdlog::error("node {:d} buffer loading failed", id_);
      node_ifs.close();
      return FAILURE;
    }
  }

  // TODO some nodes report node_ifs.good() == 0
//  assert(node_ifs.good());
  node_ifs.close();

  if (!children_.empty()) {
    for (const auto & child : children_) {
      status = child->load(ifs_buf, buffer_manager, nnode, nleaf);

      if (status == FAILURE) {
        spdlog::error("child {:d} of {:d} loading failed", child->get_id(), id_);
        node_ifs.close();
        return FAILURE;
      }
    }

    spdlog::info("subtree {:d} loaded", id_);
  } else {
    if (nseries_ == buffer_.get().size()) {
      spdlog::info("leaf {:d} loaded {:d}", id_, nseries_);

      nleaf += 1;
    } else {
      spdlog::error("leaf {:d} loaded {:d}; expected {:d}", id_, buffer_.get().size(), nseries_);
      return FAILURE;
    }
  }

  nnode += 1;
  return SUCCESS;
}
