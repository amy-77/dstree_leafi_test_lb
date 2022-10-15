//
// Created by Qitong Wang on 2022/10/10.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "eapca.h"

#include <tuple>
#include <cmath>
#include <iostream>

#include <boost/format.hpp>

#include "stat.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::EAPCA::EAPCA(const VALUE_TYPE *series_ptr,
                     ID_TYPE series_length,
                     ID_TYPE vertical_split_nsegment) :
    nvalues_(series_length),
    series_ptr_(series_ptr) {
#ifdef DEBUG
  // TODO support vertical_split_nsegment > 2
  assert(vertical_split_nsegment == 2);
#endif

  VALUE_TYPE mean = 0, mean_first_half = 0, mean_last_half = 0;
  VALUE_TYPE std = 0, std_first_half = 0, std_last_half = 0;

  ID_TYPE breakpoint = nvalues_ / 2;

  ID_TYPE value_id = 0;
  while (value_id < breakpoint) {
    mean_first_half += series_ptr[value_id];
    value_id += 1;
  }
  while (value_id < nvalues_) {
    mean_last_half += series_ptr[value_id];
    value_id += 1;
  }
  mean = (mean_first_half + mean_last_half) / static_cast<VALUE_TYPE>(nvalues_);
  mean_first_half /= static_cast<VALUE_TYPE>(breakpoint);
  mean_last_half /= static_cast<VALUE_TYPE>(nvalues_ - breakpoint);

  value_id = 0;
  while (value_id < breakpoint) {
    std_first_half += (series_ptr[value_id] - mean_first_half) * (series_ptr[value_id] - mean_first_half);
    std += (series_ptr[value_id] - mean) * (series_ptr[value_id] - mean);
    value_id += 1;
  }
  while (value_id < nvalues_) {
    std_last_half += (series_ptr[value_id] - mean_last_half) * (series_ptr[value_id] - mean_last_half);
    std += (series_ptr[value_id] - mean) * (series_ptr[value_id] - mean);
    value_id += 1;
  }

  std = sqrt(std / static_cast<VALUE_TYPE>(nvalues_));
  std_first_half = sqrt(std_first_half / static_cast<VALUE_TYPE>(nvalues_));
  std_last_half = sqrt(std_last_half / static_cast<VALUE_TYPE>(nvalues_));

  segment_means_.push_back(mean);
  segment_stds_.push_back(std);
  nsegment_ = 1;

  subsegment_means_.push_back(mean_first_half);
  subsegment_means_.push_back(mean_last_half);
  subsegment_stds_.push_back(std_first_half);
  subsegment_stds_.push_back(std_last_half);
  nsubsegment_ = 2;
}

RESPONSE dstree::EAPCA::split(const std::shared_ptr<dstree::Config> &config,
                              const std::shared_ptr<Split> &split,
                              const std::vector<ID_TYPE> &segment_lengths,
                              const std::vector<ID_TYPE> &subsegment_lengths) {
  // TODO enable non-splittable segment
  // assume all segments could be vertical split, i.e., segment_length >= config->vertical_split_nsubsegment_

  if (split->is_vertical_split_) {
    auto segment_mean_iter = segment_means_.begin();
    std::advance(segment_mean_iter, split->split_segment_id_);

    auto subsegment_mean_iter = subsegment_means_.begin();
    std::advance(subsegment_mean_iter, config->vertical_split_nsubsegment_ * split->split_segment_id_);

    auto segment_std_iter = segment_stds_.begin();
    std::advance(segment_std_iter, split->split_segment_id_);

    auto subsegment_std_iter = subsegment_stds_.begin();
    std::advance(subsegment_std_iter, config->vertical_split_nsubsegment_ * split->split_segment_id_);

    for (ID_TYPE i = 1; i < config->vertical_split_nsubsegment_; ++i) {
      segment_means_.insert(segment_mean_iter, *subsegment_mean_iter);
      subsegment_mean_iter = subsegment_means_.erase(subsegment_mean_iter);

      segment_stds_.insert(segment_std_iter, *subsegment_std_iter);
      subsegment_std_iter = subsegment_stds_.erase(subsegment_std_iter);
    }

    *segment_mean_iter = *subsegment_mean_iter;
    subsegment_mean_iter = subsegment_means_.erase(subsegment_mean_iter);

    *segment_std_iter = *subsegment_std_iter;
    subsegment_std_iter = subsegment_stds_.erase(subsegment_std_iter);

    ID_TYPE value_offset = 0;
    for (ID_TYPE i = 0; i < split->split_segment_id_; ++i) {
      value_offset += segment_lengths[i];
    }

    nsegment_ += config->vertical_split_nsubsegment_ - 1;

#ifdef DEBUG
    assert(nsegment_ == segment_means_.size());
    assert(nsegment_ == segment_stds_.size());
#endif

    VALUE_TYPE mean, std;
    ID_TYPE subsegment_offset = config->vertical_split_nsubsegment_ * split->split_segment_id_;

    for (ID_TYPE subsegment_local_id = 0;
         subsegment_local_id < config->vertical_split_nsubsegment_;
         ++subsegment_local_id) {
      ID_TYPE subsegment_length = subsegment_lengths[subsegment_offset + subsegment_local_id];
      ID_TYPE subsubsegment_length = subsegment_length / config->vertical_split_nsubsegment_;
      ID_TYPE last_subsubsegment_length =
          subsegment_length - subsubsegment_length * (config->vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
      assert(subsubsegment_length > 0);
      assert(last_subsubsegment_length > 0);
#endif

      for (ID_TYPE i = 1; i < config->vertical_split_nsubsegment_; ++i) {
        std::tie(mean, std) = upcite::cal_mean_std(series_ptr_ + value_offset, subsubsegment_length);

        subsegment_means_.insert(subsegment_mean_iter, mean);
        subsegment_stds_.insert(subsegment_std_iter, std);

        value_offset += subsubsegment_length;
      }

      std::tie(mean, std) = upcite::cal_mean_std(series_ptr_ + value_offset, last_subsubsegment_length);

      subsegment_means_.insert(subsegment_mean_iter, mean);
      subsegment_stds_.insert(subsegment_std_iter, std);

      value_offset += last_subsubsegment_length;
    }

    nsubsegment_ += (config->vertical_split_nsubsegment_ - 1) * config->vertical_split_nsubsegment_;

#ifdef DEBUG
    assert(nsubsegment_ == subsegment_means_.size());
    assert(nsubsegment_ == subsegment_stds_.size());
#endif
  }

  return SUCCESS;
}

VALUE_TYPE dstree::EAPCA::get_segment_value(ID_TYPE segment_id, bool is_mean) const {
  if (is_mean) {
    auto mean_iter = segment_means_.cbegin();
    std::advance(mean_iter, segment_id);
    return *mean_iter;
  } else {
    auto std_iter = segment_stds_.cbegin();
    std::advance(std_iter, segment_id);
    return *std_iter;
  }
}

VALUE_TYPE dstree::EAPCA::get_subsegment_value(ID_TYPE subsegment_id, bool is_mean) const {
  if (is_mean) {
    auto mean_iter = subsegment_means_.cbegin();
    std::advance(mean_iter, subsegment_id);
    return *mean_iter;
  } else {
    auto std_iter = subsegment_stds_.cbegin();
    std::advance(std_iter, subsegment_id);
    return *std_iter;
  }
}
dstree::EAPCA_Envelope::EAPCA_Envelope(const std::shared_ptr<dstree::EAPCA_Envelope> &eapca_envelope) {
  nsegment_ = eapca_envelope->nsegment_;
  nsubsegment_ = eapca_envelope->nsubsegment_;

  segment_lengths_ = eapca_envelope->segment_lengths_;
  subsegment_lengths_ = eapca_envelope->subsegment_lengths_;

  min_means_ = eapca_envelope->min_means_;
  max_means_ = eapca_envelope->max_means_;
  min_stds_ = eapca_envelope->min_stds_;
  max_stds_ = eapca_envelope->max_stds_;

  subsegment_min_means_ = eapca_envelope->subsegment_min_means_;
  subsegment_max_means_ = eapca_envelope->subsegment_max_means_;
  subsegment_min_stds_ = eapca_envelope->subsegment_min_stds_;
  subsegment_max_stds_ = eapca_envelope->subsegment_max_stds_;
}

dstree::EAPCA_Envelope::EAPCA_Envelope(const std::shared_ptr<Config> &config,
                                       ID_TYPE nsegment) {
  nsegment_ = nsegment;
  nsubsegment_ = nsegment_ * config->vertical_split_nsubsegment_;

  ID_TYPE series_length = config->series_length_;
  ID_TYPE segment_length = series_length / nsegment_;

#ifdef DEBUG
  assert(segment_length > config->vertical_split_nsubsegment_);
#endif

  segment_lengths_.assign(nsegment_ - 1, segment_length);

  ID_TYPE subsegment_length = segment_length / config->vertical_split_nsubsegment_;
  ID_TYPE last_subsegment_length = segment_length - subsegment_length * (config->vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
  assert(subsegment_length > 0);
  assert(last_subsegment_length > 0);
#endif

  for (ID_TYPE segment_id = 0; segment_id < nsegment_ - 1; ++segment_id) {
    subsegment_lengths_.insert(subsegment_lengths_.end(), config->vertical_split_nsubsegment_ - 1, subsegment_length);
    subsegment_lengths_.push_back(last_subsegment_length);
  }

  ID_TYPE last_segment_length = series_length - segment_length * (nsegment_ - 1);

#ifdef DEBUG
  assert(last_segment_length > config->vertical_split_nsubsegment_);
#endif

  segment_lengths_.push_back(last_segment_length);

  subsegment_length = last_segment_length / config->vertical_split_nsubsegment_;
  last_subsegment_length = last_segment_length - subsegment_length * (config->vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
  assert(subsegment_length > 0);
  assert(last_subsegment_length > 0);
#endif

  subsegment_lengths_.insert(subsegment_lengths_.end(), config->vertical_split_nsubsegment_ - 1, subsegment_length);
  subsegment_lengths_.push_back(last_subsegment_length);

#ifdef DEBUG
  assert(nsubsegment_ == subsegment_lengths_.size());
#endif

  initialize_stats();
}

dstree::EAPCA_Envelope::EAPCA_Envelope(const std::shared_ptr<Config> &config,
                                       const std::shared_ptr<dstree::EAPCA_Envelope> &parent_eapca_envelope,
                                       const std::shared_ptr<dstree::Split> &parent_split,
                                       const std::shared_ptr<upcite::Logger> &logger) {
  if (parent_split->is_vertical_split_) {
    nsegment_ = parent_eapca_envelope->nsegment_ + config->vertical_split_nsubsegment_ - 1;
    nsubsegment_ = parent_eapca_envelope->nsubsegment_ +
        +config->vertical_split_nsubsegment_ * (config->vertical_split_nsubsegment_ - 1);

    for (ID_TYPE segment_id = 0; segment_id < parent_eapca_envelope->nsegment_; ++segment_id) {
      if (segment_id == parent_split->split_segment_id_) {
        for (ID_TYPE subsegment_id = segment_id * config->vertical_split_nsubsegment_;
             subsegment_id < (segment_id + 1) * config->vertical_split_nsubsegment_;
             ++subsegment_id) {
          segment_lengths_.push_back(parent_eapca_envelope->subsegment_lengths_[subsegment_id]);

          ID_TYPE subsegment_length = parent_eapca_envelope->subsegment_lengths_[subsegment_id];
          ID_TYPE subsubsegment_length = subsegment_length / config->vertical_split_nsubsegment_;
          ID_TYPE last_subsubsegment_length =
              subsegment_length - subsubsegment_length * (config->vertical_split_nsubsegment_ - 1);

#ifdef DEBUG
          assert(subsubsegment_length > 0);
          assert(last_subsubsegment_length > 0);
#endif

          subsegment_lengths_.insert(subsegment_lengths_.end(),
                                     config->vertical_split_nsubsegment_ - 1,
                                     subsubsegment_length);
          subsegment_lengths_.push_back(last_subsubsegment_length);
        }
      } else {
        segment_lengths_.push_back(parent_eapca_envelope->segment_lengths_[segment_id]);

        for (ID_TYPE subsegment_id = segment_id * config->vertical_split_nsubsegment_;
             subsegment_id < (segment_id + 1) * config->vertical_split_nsubsegment_;
             ++subsegment_id) {
          subsegment_lengths_.push_back(parent_eapca_envelope->subsegment_lengths_[subsegment_id]);
        }
      }
    }
  } else {
    nsegment_ = parent_eapca_envelope->nsegment_;
    nsubsegment_ = parent_eapca_envelope->nsubsegment_;

    segment_lengths_ = parent_eapca_envelope->segment_lengths_;
    subsegment_lengths_ = parent_eapca_envelope->subsegment_lengths_;
  }

#ifdef DEBUG
  if (logger != nullptr) {
    MALAT_LOG(logger->logger, trivial::debug) << boost::format("nsegment_ = %d, segment_lengths_.size() = %d")
          % nsegment_ % segment_lengths_.size();
    MALAT_LOG(logger->logger, trivial::debug) << boost::format("nsubsegment_ = %d, subsegment_lengths_.size() = %d")
          % nsubsegment_ % subsegment_lengths_.size();
  }

  assert(nsegment_ == segment_lengths_.size());
  assert(nsubsegment_ == subsegment_lengths_.size());
#endif

  initialize_stats();
}

RESPONSE dstree::EAPCA_Envelope::initialize_stats() {
  min_means_.assign(nsegment_, constant::MAX_VALUE);
  max_means_.assign(nsegment_, constant::MIN_VALUE);
  min_stds_.assign(nsegment_, constant::MAX_VALUE);
  max_stds_.assign(nsegment_, constant::MIN_VALUE);

  subsegment_min_means_.assign(nsubsegment_, constant::MAX_VALUE);
  subsegment_max_means_.assign(nsubsegment_, constant::MIN_VALUE);
  subsegment_min_stds_.assign(nsubsegment_, constant::MAX_VALUE);
  subsegment_max_stds_.assign(nsubsegment_, constant::MIN_VALUE);

  return SUCCESS;
}

RESPONSE dstree::EAPCA_Envelope::update(const std::shared_ptr<dstree::EAPCA> &series_eapca) {
  if (nsegment_ == series_eapca->nsegment_ && nsubsegment_ == series_eapca->nsubsegment_) {
    auto mean_iter = series_eapca->segment_means_.cbegin();
    auto std_iter = series_eapca->segment_stds_.cbegin();

    for (ID_TYPE i = 0; i < series_eapca->nsegment_; ++i, ++mean_iter, ++std_iter) {
      if (min_means_[i] > *mean_iter) {
        min_means_[i] = *mean_iter;
      } else if (max_means_[i] < *mean_iter) {
        max_means_[i] = *mean_iter;
      }

      if (min_stds_[i] > *std_iter) {
        min_stds_[i] = *std_iter;
      } else if (max_stds_[i] < *std_iter) {
        max_stds_[i] = *std_iter;
      }
    }

    mean_iter = series_eapca->subsegment_means_.cbegin();
    std_iter = series_eapca->subsegment_stds_.cbegin();

    for (ID_TYPE i = 0; i < series_eapca->nsubsegment_; ++i, ++mean_iter, ++std_iter) {
      if (subsegment_min_means_[i] > *mean_iter) {
        subsegment_min_means_[i] = *mean_iter;
      } else if (subsegment_max_means_[i] < *mean_iter) {
        subsegment_max_means_[i] = *mean_iter;
      }

      if (subsegment_min_stds_[i] > *std_iter) {
        subsegment_min_stds_[i] = *std_iter;
      } else if (subsegment_max_stds_[i] < *std_iter) {
        subsegment_max_stds_[i] = *std_iter;
      }
    }

    return SUCCESS;
  }

  return FAILURE;
}
