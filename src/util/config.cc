//
// Created by Qitong Wang on 2022/10/2.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "config.h"

#include <iostream>

//#include <boost/format.hpp>
#include <boost/filesystem.hpp>
//#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <spdlog/spdlog.h>

//#include "logger.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
namespace constant = upcite::constant;

namespace dstree = upcite::dstree;

dstree::Config::Config(int argc, char *argv[]) :
    db_nseries_(-1),
    query_nseries_(-1),
    series_length_(-1),
    is_znormalized_(true),
    leaf_max_nseries_(10000),
    batch_load_nseries_(-1),
    default_nbuffer_(1024 * 64),
    on_disk_(false),
    index_persist_folderpath_("."),
    index_persist_file_postfix_(".bin"),
    node_nchild_(2),
    vertical_split_nsubsegment_(2),
    vertical_split_gain_tradeoff_factor_(2),
    is_exact_search_(false),
    search_max_nseries_(-1),
    search_max_nnode_(-1),
    n_nearest_neighbor_(1),
    examine_ground_truth_(false),
    require_neurofilter_(false),
    filter_dim_latent_(-1),
    filter_train_dropout_p_(0.5),
    filter_leaky_relu_negative_slope_(0.1),
    filter_train_is_gpu_(false),
    filter_infer_is_gpu_(false),
    filter_device_id_(0),
    filter_train_nexample_(-1),
    filter_train_batchsize_(-1),
    filter_train_nepoch_(100),
    filter_train_learning_rate_(0.01),
    filter_train_min_lr_(0.0001),
    filter_train_clip_grad_norm_type_(2),
    filter_train_clip_grad_max_norm_(1),
    filter_train_is_mthread_(false),
    filter_collect_nthread_(-1),
    filter_train_nthread_(4),
    filter_remove_square_(false),
    filter_train_val_split_(0.9) {
  po::options_description po_desc("DSTree C++ implementation. Copyright (c) 2022 UPCité.");

  po_desc.add_options()
      ("help", "produce help message")
      ("log_filepath", po::value<std::string>(&log_filepath_)->default_value("./dstree.log"),
       "Logging file path")
      ("db_filepath", po::value<std::string>(&db_filepath_)->required(),
       "Database file path")
      ("query_filepath", po::value<std::string>(&query_filepath_)->required(),
       "Query file path")
      ("series_length", po::value<ID_TYPE>(&series_length_)->required(),
       "Series length")
      ("is_znormalized", po::bool_switch(&is_znormalized_)->default_value(false),
       "Whether data has been z-normalized")
      ("db_size", po::value<ID_TYPE>(&db_nseries_)->required(),
       "Number of series in database")
      ("query_size", po::value<ID_TYPE>(&query_nseries_)->required(),
       "Number of query series")
      ("leaf_size", po::value<ID_TYPE>(&leaf_max_nseries_)->default_value(10000),
       "Maximal leaf node size")
      ("batch_load_size", po::value<ID_TYPE>(&batch_load_nseries_),
       "Maximal number of series for batch loading")
      ("default_nbuffer", po::value<ID_TYPE>(&default_nbuffer_),
       "Default number of node buffers (will increase when needed)")
      ("on_disk", po::bool_switch(&on_disk_)->default_value(false),
       "Whether to build an on-disk index (otherwise an in-memory index)")
      ("index_persist_folderpath", po::value<std::string>(&index_persist_folderpath_),
       "Index on-disk root folderpath")
      ("node_nchildren", po::value<ID_TYPE>(&node_nchild_)->default_value(2),
       "Number of child nodes (i.e., fanout) for each parent node")
      ("vsplit_nsubsegment", po::value<ID_TYPE>(&vertical_split_nsubsegment_)->default_value(2),
       "Number of subsegments for vertical split")
      ("vsplit_gain_factor", po::value<VALUE_TYPE>(&vertical_split_gain_tradeoff_factor_)->default_value(2),
       "Increase factor of vertical splits versus horizontal splits")
      ("exact_search", po::bool_switch(&is_exact_search_)->default_value(false),
       "Whether to conduct exact search (or approximate search)")
      ("search_max_nseries", po::value<ID_TYPE>(&search_max_nseries_)->default_value(-1),
       "Maximal number of series to be checked during query answering")
      ("search_max_nnode", po::value<ID_TYPE>(&search_max_nnode_)->default_value(-1),
       "Maximal number of nodes to be checked during query answering")
      ("n_nearest_neighbor", po::value<ID_TYPE>(&n_nearest_neighbor_)->default_value(1),
       "Number of nearest neighbors to be returned")
      ("ground_truth", po::bool_switch(&examine_ground_truth_)->default_value(false),
       "Whether to fetch the ground truths, i.e., linear scan without pruning")
      ("require_neurofilter", po::bool_switch(&require_neurofilter_)->default_value(false),
       "Whether to implant neurofilters")
      ("dim_latent", po::value<ID_TYPE>(&filter_dim_latent_),
       "Dimension of neural model latent variables")
      ("dropout_p", po::value<VALUE_TYPE>(&filter_train_dropout_p_)->default_value(0.5),
       "Dropout probability for MLP latent layer")
      ("leaky_relu_negative_slope", po::value<VALUE_TYPE>(&filter_leaky_relu_negative_slope_)->default_value(0.1),
       "Leaky ReLU negative slope for MLP")
      ("filter_train_is_gpu", po::bool_switch(&filter_train_is_gpu_)->default_value(false),
       "Whether to train neurofilters on GPU (other on CPU)")
      ("filter_infer_is_gpu", po::bool_switch(&filter_infer_is_gpu_)->default_value(false),
       "Whether to run neurofilters on GPU (other on CPU)")
      ("device_id", po::value<ID_TYPE>(&filter_device_id_)->default_value(0),
       "GPU device id")
      ("filter_train_nexample", po::value<ID_TYPE>(&filter_train_nexample_),
       "Number of train examples for neurofilters")
      ("filter_train_batchsize", po::value<ID_TYPE>(&filter_train_batchsize_)->default_value(-1),
       "Neurofilter train batch size")
      ("filter_train_nepoch", po::value<ID_TYPE>(&filter_train_nepoch_)->default_value(100),
       "Neurofilter train (maximal) number of epochs")
      ("learning_rate", po::value<VALUE_TYPE>(&filter_train_learning_rate_)->default_value(0.01),
       "Neurofilter train learning rate")
      ("filter_train_min_lr", po::value<VALUE_TYPE>(&filter_train_min_lr_)->default_value(0.0001),
       "Neurofilter train minimal learning rate, for adjusting learning rates")
      ("clip_grad_norm_type", po::value<VALUE_TYPE>(&filter_train_clip_grad_norm_type_)->default_value(2),
       "Gradient clipping norm type")
      ("clip_grad_max_norm", po::value<VALUE_TYPE>(&filter_train_clip_grad_max_norm_)->default_value(1),
       "Gradient clipping max norm")
      ("filter_query_filepath", po::value<std::string>(&filter_query_filepath_),
       "Query file path to train neurofilters")
      ("filter_train_mthread", po::bool_switch(&filter_train_is_mthread_)->default_value(false),
       "Whether to train neurofilters multithreadingly")
      ("filter_collect_nthread", po::value<ID_TYPE>(&filter_collect_nthread_)->default_value(-1),
       "Number of threads to collect neurofilter train set; default filter_train_nthread")
      ("filter_train_nthread", po::value<ID_TYPE>(&filter_train_nthread_)->default_value(4),
       "Number of threads to train neurofilters")
      ("filter_remove_square", po::bool_switch(&filter_remove_square_)->default_value(false),
       "Whether to use real distance (instead of square distances) to train filters")
      ("filter_train_val_split", po::value<VALUE_TYPE>(&filter_train_val_split_)->default_value(0.9),
       "Neurofilter train train/val split ratio");

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, po_desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << po_desc << std::endl;
    exit(0);
  }

  if (vm.count("index_persist_folderpath")) {
    index_persist_folderpath_ = fs::system_complete(index_persist_folderpath_).string();
  } else {
    index_persist_folderpath_ = fs::system_complete(fs::current_path()).string();
  }

  if (on_disk_) {
    if (!fs::is_directory(index_persist_folderpath_)) {
      fs::create_directory(index_persist_folderpath_);
    }

    if (!boost::algorithm::ends_with(index_persist_folderpath_, "/")) {
      index_persist_folderpath_ += "/";
    }
  }

  if (batch_load_nseries_ < 0) {
    batch_load_nseries_ = db_nseries_;
  }

  if (vertical_split_nsubsegment_ != 2) {
    if (vm.count("vsplit_gain_factor") == 0) {
      vertical_split_gain_tradeoff_factor_ = static_cast<VALUE_TYPE>(vertical_split_nsubsegment_);
    }
  }

  if (search_max_nseries_ > 0 || search_max_nnode_ > 0) {
    is_exact_search_ = true;
  }

  if (is_exact_search_) {
    if (search_max_nseries_ < 1) {
      search_max_nseries_ = db_nseries_;
    }

    if (search_max_nnode_ < 1) {
      search_max_nnode_ = db_nseries_;
    }
  }

  if (require_neurofilter_) {
    if (filter_dim_latent_ < 0) {
      filter_dim_latent_ = series_length_;
    }

    if (filter_train_nexample_ < 0) {
      std::cout << "Please specify the number of neurofilter train examples by setting --neurofilter_train_nexample"
                << std::endl;
      exit(-1);
    }

    if (filter_train_batchsize_ < 0) {
      filter_train_batchsize_ = filter_train_nexample_;
    }

    if (vm.count("filter_infer_is_gpu") < 1) {
      filter_infer_is_gpu_ = filter_train_is_gpu_;
    }

    if (vm.count("dim_latent") < 1) {
      filter_dim_latent_ = series_length_;
    }

    if (filter_train_is_mthread_) {
      if (filter_collect_nthread_ < 0) {
        filter_collect_nthread_ = filter_train_nthread_;
      }
    }
  }
}

void dstree::Config::log() {
  spdlog::info("db_filepath = {:s}", db_filepath_);
  spdlog::info("query_filepath = {:s}", query_filepath_);
  spdlog::info("is_znormalized = {:b}", is_znormalized_);

  spdlog::info("db_nseries = {:d}", db_nseries_);
  spdlog::info("query_nseries = {:d}", query_nseries_);
  spdlog::info("leaf_max_nseries = {:d}", leaf_max_nseries_);

  spdlog::info("batch_load_nseries = {:d}", batch_load_nseries_);
  spdlog::info("default_nbuffer = {:d}", default_nbuffer_);

  spdlog::info("on_disk = {:b}", on_disk_);
  spdlog::info("index_persist_folderpath = {:s}", index_persist_folderpath_);
  spdlog::info("index_persist_file_postfix = {:s}", index_persist_file_postfix_);

  spdlog::info("node_nchild = {:d}", node_nchild_);
  spdlog::info("vertical_split_nsubsegment = {:d}", vertical_split_nsubsegment_);
  spdlog::info("vertical_split_gain_tradeoff_factor = {:.3f}", vertical_split_gain_tradeoff_factor_);

  spdlog::info("is_exact_search = {:b}", is_exact_search_);
  spdlog::info("search_max_nseries = {:d}", search_max_nseries_);
  spdlog::info("search_max_nnode = {:d}", search_max_nnode_);

  spdlog::info("n_nearest_neighbor = {:d}", n_nearest_neighbor_);

  spdlog::info("examine_ground_truth = {:d}", examine_ground_truth_);

  spdlog::info("require_neurofilter = {:b}", require_neurofilter_);
  spdlog::info("filter_dim_latent = {:d}", filter_dim_latent_);
  spdlog::info("filter_leaky_relu_negative_slope = {:.3f}", filter_leaky_relu_negative_slope_);
  spdlog::info("filter_train_dropout_p_ = {:.3f}", filter_train_dropout_p_);

  spdlog::info("filter_train_is_gpu_ = {:b}", filter_train_is_gpu_);
  spdlog::info("filter_infer_is_gpu = {:b}", filter_infer_is_gpu_);
  spdlog::info("filter_device_id = {:d}", filter_device_id_);

  spdlog::info("filter_train_nexample = {:d}", filter_train_nexample_);
  spdlog::info("filter_train_batchsize = {:d}", filter_train_batchsize_);
  spdlog::info("filter_train_nepoch = {:d}", filter_train_nepoch_);
  spdlog::info("filter_train_learning_rate = {:.3f}", filter_train_learning_rate_);
  spdlog::info("filter_train_min_lr = {:.3f}", filter_train_min_lr_);

  spdlog::info("filter_train_clip_grad_norm_type = {:.1f}", filter_train_clip_grad_norm_type_);
  spdlog::info("filter_train_clip_grad_max_norm = {:.3f}", filter_train_clip_grad_max_norm_);

  spdlog::info("filter_query_filepath = {:s}", filter_query_filepath_);

  spdlog::info("filter_train_is_mthread = {:b}", filter_train_is_mthread_);
  spdlog::info("filter_collect_nthread = {:d}", filter_collect_nthread_);
  spdlog::info("filter_train_nthread = {:d}", filter_train_nthread_);

  spdlog::info("filter_remove_square = {:b}", filter_remove_square_);

  spdlog::info("filter_train_val_split = {:.3f}", filter_train_val_split_);
}
