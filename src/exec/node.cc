//
// Created by Qitong Wang on 2022/10/4.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "node.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <memory>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

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
    filter_(nullptr) {
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
    filter_(nullptr) {
  split_ = std::make_unique<dstree::Split>();
  children_.reserve(config.node_nchild_);

  eapca_envelope_ = std::make_unique<EAPCAEnvelope>(eapca_envelope);
}

dstree::Node &dstree::Node::route(dstree::EAPCA &series_eapca, bool is_update_statistics) {
  ID_TYPE target_child_id;

  if (is_update_statistics) {
    eapca_envelope_->update(series_eapca);
  }

  if (split_->is_vertical_split_) {
    target_child_id = split_->route(series_eapca.get_subsegment_value(
        split_->split_subsegment_id_, split_->horizontal_split_mode_ == MEAN));

    series_eapca.split(config_, *split_, eapca_envelope_->segment_lengths_, eapca_envelope_->subsegment_lengths_);
  } else {
    target_child_id = split_->route(series_eapca.get_segment_value(
        split_->split_segment_id_, split_->horizontal_split_mode_ == MEAN));
  }

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

  response = static_cast<RESPONSE>(response || eapca_envelope_->update(series_eapca));

  if (response == SUCCESS) {
    nseries_ += 1;
  } else {}

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

    //
    // check horizontal split with mean
    current_split->horizontal_split_mode_ = MEAN;
    mean_width_children = mean_width / nchild;
    range_children = 0;

    for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
      range_children += segment_length * (mean_width_children * mean_width_children + max_std * max_std);
    }

    quality_gain = range_parent - range_children / nchild;

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
    }

    //
    // check horizontal split with std
    current_split->horizontal_split_mode_ = STD;
    std_width_children = (max_std - eapca_envelope_->segment_min_stds_[segment_id]) / nchild;
    range_children = 0;

    for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
      max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
      range_children += segment_length * (mean_width * mean_width + max_std_child * max_std_child);
    }

    quality_gain = range_parent - range_children / nchild;

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
    }

    current_split->is_vertical_split_ = true;

    for (subsegment_id = segment_id * config_.get().vertical_split_nsubsegment_;
         subsegment_id < (segment_id + 1) * config_.get().vertical_split_nsubsegment_;
         ++subsegment_id) {
      current_split->split_subsegment_id_ = subsegment_id;

      mean_width = eapca_envelope_->subsegment_max_means_[subsegment_id]
          - eapca_envelope_->subsegment_min_means_[subsegment_id];
      max_std = eapca_envelope_->subsegment_max_stds_[subsegment_id];
      subsegment_length = static_cast<VALUE_TYPE>(eapca_envelope_->subsegment_lengths_[subsegment_id]);

      range_parent = subsegment_length * (mean_width * mean_width + max_std * max_std);

      //
      // check vertical split with mean
      current_split->horizontal_split_mode_ = MEAN;
      mean_width_children = mean_width / nchild;
      range_children = 0;

      for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
        range_children += subsegment_length * (mean_width_children * mean_width_children + max_std * max_std);
      }

      quality_gain = range_parent - range_children / nchild;

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
      }

      //
      // check vertical split with std
      current_split->horizontal_split_mode_ = STD;
      std_width_children = (max_std - eapca_envelope_->subsegment_min_stds_[subsegment_id]) / nchild;
      range_children = 0;

      for (child_id = 0; child_id < config_.get().node_nchild_; ++child_id) {
        max_std_child = max_std - std_width_children * static_cast<VALUE_TYPE>(child_id);
        range_children += subsegment_length * (mean_width * mean_width + max_std_child * max_std_child);
      }

      quality_gain = range_parent - range_children / nchild;

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
      }
    }
  }

  //
  // update split info
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
  }

  // TODO support index building in batch: flushed data and cached data should be combined
  for (ID_TYPE node_series_id = 0; node_series_id < buffer_.get().size(); ++node_series_id) {
    ID_TYPE series_batch_id = buffer_.get().get_offset(node_series_id);

    // statistics was already updated when they insert into the parent node
    dstree::Node &target_child = route(buffer_manager.get_series_eapca(series_batch_id), false);
    target_child.insert(series_batch_id, buffer_manager.get_series_eapca(series_batch_id));
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

  buffer_.get().clean(true);
  nseries_ = 0;

  return SUCCESS;
}

RESPONSE dstree::Node::search(const VALUE_TYPE *query_series_ptr,
                              dstree::Answers &answer,
                              ID_TYPE &visited_node_counter,
                              ID_TYPE &visited_series_counter) const {
  const VALUE_TYPE *db_series_ptr = buffer_.get().get_first_series_ptr();


  while (db_series_ptr != nullptr) {
    VALUE_TYPE distance = upcite::cal_EDsquare(db_series_ptr, query_series_ptr, config_.get().series_length_);


    if (answer.is_bsf(distance)) {
      answer.push_bsf(distance, id_);

      spdlog::info("query {:d} update bsf {:.3f} at node {:d} visiting node {:d} series {:d}",
                   answer.query_id_, distance, id_,
                   visited_node_counter, visited_series_counter);
    }

    visited_series_counter += 1;
    db_series_ptr = buffer_.get().get_next_series_ptr();
  }

  buffer_.get().reset(true, true);

  visited_node_counter += 1;

  return SUCCESS;
}




VALUE_TYPE dstree::Node::search(const VALUE_TYPE *query_series_ptr,
                                VALUE_TYPE *m256_fetch_cache,
                                VALUE_TYPE bsf_distance) const {
  const VALUE_TYPE *db_series_ptr = buffer_.get().get_first_series_ptr();
  VALUE_TYPE local_bsf = constant::MAX_VALUE;

  while (db_series_ptr != nullptr) {
    VALUE_TYPE distance = upcite::cal_EDsquare(db_series_ptr, query_series_ptr, config_.get().series_length_);


    if (distance < local_bsf) {
      local_bsf = distance;
    }

    db_series_ptr = buffer_.get().get_next_series_ptr();
  }

  buffer_.get().reset(true, true);

  return local_bsf;
}




VALUE_TYPE dstree::Node::search_mt(const VALUE_TYPE *query_series_ptr,
                                   Answers &answer,
                                   pthread_mutex_t *answer_mutex) const {
  const VALUE_TYPE *db_series_ptr = buffer_.get().get_first_series_ptr();
  VALUE_TYPE local_bsf = constant::MAX_VALUE;

  pthread_mutex_lock(answer_mutex);
  VALUE_TYPE global_bsf = answer.get_bsf();
  pthread_mutex_unlock(answer_mutex);

  while (db_series_ptr != nullptr) {
    VALUE_TYPE distance = upcite::cal_EDsquare(db_series_ptr, query_series_ptr, config_.get().series_length_);

    if (distance < local_bsf) {
      local_bsf = distance;

      if (local_bsf < global_bsf) {
        pthread_mutex_lock(answer_mutex);
        answer.check_push_bsf(local_bsf, id_);

        global_bsf = answer.get_bsf();
        pthread_mutex_unlock(answer_mutex);

#ifdef DEBUG
        spdlog::info("query {:d} update bsf {:.3f} at node {:d}",
                     answer.query_id_, local_bsf, id_);
#endif
      }
    }

    db_series_ptr = buffer_.get().get_next_series_ptr();
  }

  buffer_.get().reset(true, true);

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
  assert(node_ofs.is_open());

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

//  spdlog::debug("node {:d} buffer dump {:s} load {:s}",
//                id_, buffer_.get().dump_filepath_, buffer_.get().load_filepath_);
  buffer_.get().dump(node_ofs);

  if (filter_ != nullptr) {
    assert(filter_.get()->get_id() >= 0);
    ofs_id_buf[0] = filter_.get()->get_id();
  } else {
    ofs_id_buf[0] = -1;
  }
  node_ofs.write(reinterpret_cast<char *>(ofs_id_buf), sizeof(ID_TYPE));

  if (filter_ != nullptr) {
    filter_->dump(node_ofs);
  }

//  if (filter_ != nullptr) {
//    spdlog::debug("dump node {:d} filter {:d} size {:d} {:d} cache {:d}",
//                  id_, filter_.get()->get_id(), nseries_, buffer_.get().size(), buffer_.get().cached_size_);
//  } else {
//    spdlog::debug("dump node {:d} size {:d} {:d} cache {:d}",
//                  id_, nseries_, buffer_.get().size(), buffer_.get().cached_size_);
//  }

  assert(node_ofs.good());
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
  assert(node_ifs.is_open());

  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);

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

  status = buffer_.get().load(node_ifs, ifs_buf);
  assert(nseries_ == buffer_.get().size());

  if (status == FAILURE) {
    spdlog::error("node {:d} buffer loading failed", id_);
    node_ifs.close();
    return FAILURE;
  }

  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE filter_id = ifs_id_buf[0];

  if (filter_id >= 0) {
    filter_ = std::make_unique<dstree::Filter>(config_, filter_id, constant::TENSOR_PLACEHOLDER_REF);
    status = filter_->load(node_ifs, ifs_buf);

    if (status == FAILURE) {
      spdlog::error("node {:d} neurofilter loading failed", id_);
      node_ifs.close();
      return FAILURE;
    }
  }

//  if (filter_ != nullptr) {
//    spdlog::debug("load node {:d} filter {:d} or {:d} (trained {:b}) size {:d} {:d}",
//                  id_, filter_.get()->get_id(), filter_id, filter_.get()->is_trained(), nseries_, buffer_.get().size());
//  } else {
//    spdlog::debug("load node {:d} size {:d} {:d}",
//                  id_, nseries_, buffer_.get().size());
//  }

  // TODO some nodes report node_ifs.good() == 0
  assert(node_ifs.good());
  node_ifs.close();

  if (!children_.empty()) {
    for (const auto &child : children_) {
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
      if (config_.get().on_disk_) {
        spdlog::info("loaded leaf {:d}, {:d} series on disk", id_, nseries_);
      } else {
        spdlog::info("loaded leaf {:d}, {:d} series in memory", id_, nseries_);
      }

      nleaf += 1;
    } else {
      spdlog::error("loaded leaf {:d}, but {:d} series in buffer; expected {:d}", id_, buffer_.get().size(), nseries_);
      return FAILURE;
    }
  }

  nnode += 1;
  return SUCCESS;
}

ID_TYPE dstree::Node::get_num_synthetic_queries(ID_TYPE node_size_threshold) {
  ID_TYPE num_synthetic_queries = 0;

  if (is_leaf()) {
    if (nseries_ > node_size_threshold) {
      num_synthetic_queries = config_.get().filter_num_synthetic_query_per_filter_;
    }
  } else {
    for (auto &child : children_) {
      num_synthetic_queries += child->get_num_synthetic_queries(node_size_threshold);
    }
  }

  return num_synthetic_queries;
}

RESPONSE dstree::Node::synthesize_query(VALUE_TYPE *generated_queries,
                                        ID_TYPE &num_generated_queries,
                                        ID_TYPE node_size_threshold) {
  if (is_leaf()) {
    if (get_size() > node_size_threshold) {
      // TODO support multiple samples
      assert(config_.get().filter_num_synthetic_query_per_filter_ == 1);

      // provided by ChatGPT
      const gsl_rng_type *T;
      gsl_rng *r;
      gsl_rng_env_setup(); // Create a generator chosen by the environment variable GSL_RNG_TYPE
      T = gsl_rng_default;
      r = gsl_rng_alloc(T);

      // TODO support multiple samples
      bool is_succeeded = false;
      while (!is_succeeded) {
        ID_TYPE sample_series_id = upcite::get_random_int_in_range(0, nseries_);
        const VALUE_TYPE *sample_series = buffer_.get().get_series_ptr_by_id(sample_series_id);
        assert(sample_series != nullptr);

        VALUE_TYPE *generated_series = generated_queries + config_.get().series_length_ * num_generated_queries;
        memcpy(generated_series, sample_series, sizeof(VALUE_TYPE) * config_.get().series_length_);

        for (ID_TYPE value_i = 0; value_i < config_.get().series_length_; ++value_i) {
          generated_series[value_i] += (VALUE_TYPE) gsl_ran_gaussian(r, config_.get().filter_noise_level_);
        }

        RESPONSE return_code = upcite::znormalize(generated_series, config_.get().series_length_);
        if (return_code == SUCCESS) {
          num_generated_queries += 1;
          is_succeeded = true;
        } else {
          spdlog::error("filter generate broken synthetic series; regenerate");
        }
      }
    }
  } else {
    for (auto &child : children_) {
      child->synthesize_query(generated_queries, num_generated_queries, node_size_threshold);
    }
  }

  return SUCCESS;
}
