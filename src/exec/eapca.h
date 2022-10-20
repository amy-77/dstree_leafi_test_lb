//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_EAPCA_H
#define DSTREE_EAPCA_H

#include <vector>
#include <list>
#include <memory>

#include "global.h"
#include "config.h"
#include "split.h"

namespace upcite {
namespace dstree {

class EAPCA {
 public:
  EAPCA(const VALUE_TYPE *series_ptr,
        ID_TYPE series_length,
        ID_TYPE vertical_split_nsegment);
  ~EAPCA() = default;

  VALUE_TYPE get_segment_value(ID_TYPE segment_id,
                               bool is_mean = true) const;
  VALUE_TYPE get_subsegment_value(ID_TYPE subsegment_id,
                                  bool is_mean = true) const;

  RESPONSE split(const std::shared_ptr<Config> &config,
                 const std::shared_ptr<Split> &split,
                 const std::vector<ID_TYPE> &segment_lengths,
                 const std::vector<ID_TYPE> &subsegment_lengths);

  ID_TYPE nsegment_, nsubsegment_;
  std::list<VALUE_TYPE> segment_means_, segment_stds_;
  std::list<VALUE_TYPE> subsegment_means_, subsegment_stds_;

 private:
  ID_TYPE nvalues_;
  const VALUE_TYPE *series_ptr_;
};

class EAPCAEnvelope {
 public:
  explicit EAPCAEnvelope(const std::shared_ptr<EAPCAEnvelope> &eapca_envelope);
  EAPCAEnvelope(const std::shared_ptr<Config> &config,
                ID_TYPE nsegment);
  EAPCAEnvelope(const std::shared_ptr<Config> &config,
                const std::shared_ptr<EAPCAEnvelope> &parent_eapca_envelope,
                const std::shared_ptr<Split> &parent_split,
                const std::shared_ptr<upcite::Logger> &logger = nullptr);
  ~EAPCAEnvelope() = default;

  RESPONSE update(const std::shared_ptr<dstree::EAPCA> &series_eapca);

  VALUE_TYPE cal_lower_bound_EDsquare(const VALUE_TYPE *series_ptr,
                                      const std::shared_ptr<upcite::Logger>& logger) const;
  VALUE_TYPE cal_upper_bound_EDsquare(const VALUE_TYPE *series_ptr) const;

  EAPCAEnvelope &operator=(const EAPCAEnvelope &eapca_envelope) = default;

  ID_TYPE nsegment_, nsubsegment_;
  std::vector<ID_TYPE> segment_lengths_, subsegment_lengths_;

  std::vector<VALUE_TYPE> segment_min_means_, segment_max_means_, segment_min_stds_, segment_max_stds_;
  std::vector<VALUE_TYPE> subsegment_min_means_, subsegment_max_means_, subsegment_min_stds_, subsegment_max_stds_;

 private:
  RESPONSE initialize_stats();
};

}
}

#endif //DSTREE_EAPCA_H
