//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_CONFIG_H
#define DSTREE_CONFIG_H

#include <string>
#include <memory>

#include "global.h"

#include "logger.h"

namespace upcite {
namespace dstree {

class Config {
 public:
  Config(int argc, char *argv[]);
  ~Config() = default;

  void log();

  std::string log_filepath_;

  std::string db_filepath_;
  std::string query_filepath_;

  ID_TYPE series_length_;
  bool is_znormalized_;

  ID_TYPE db_nseries_;
  ID_TYPE query_nseries_;
  ID_TYPE leaf_max_nseries_;

  ID_TYPE batch_load_nseries_;
  ID_TYPE default_nbuffer_;

  bool on_disk_;
  std::string index_persist_folderpath_;
  std::string index_persist_file_postfix_;

  ID_TYPE node_nchild_;
  ID_TYPE vertical_split_nsubsegment_;
  VALUE_TYPE vertical_split_gain_tradeoff_factor_;

  bool is_exact_search_;
  ID_TYPE search_max_nseries_;
  ID_TYPE search_max_nnode_;

  ID_TYPE n_nearest_neighbor_;

  bool examine_ground_truth_;

  bool require_neurofilter_;
  ID_TYPE filter_dim_latent_;
  VALUE_TYPE filter_leaky_relu_negative_slope_;
  VALUE_TYPE filter_train_dropout_p_;
  bool filter_train_is_gpu_;
  bool filter_infer_is_gpu_;
  ID_TYPE filter_device_id_;
  ID_TYPE filter_train_nexample_;
  ID_TYPE filter_train_batchsize_;
  ID_TYPE filter_train_nepoch_;
  VALUE_TYPE filter_train_learning_rate_;
  VALUE_TYPE filter_train_min_lr_;
  bool filter_train_clip_grad_;
  VALUE_TYPE filter_train_clip_grad_norm_type_;
  VALUE_TYPE filter_train_clip_grad_max_norm_;

  std::string filter_query_filepath_;

  bool filter_train_is_mthread_;
  ID_TYPE filter_collect_nthread_;
  ID_TYPE filter_train_nthread_;

  bool filter_remove_square_;
  VALUE_TYPE filter_train_val_split_;
};

}
}

#endif //DSTREE_CONFIG_H
