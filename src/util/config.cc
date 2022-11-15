//
// Created by Qitong Wang on 2022/10/2.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "config.h"

#include <iostream>

#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "logger.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;

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
    is_ground_truth_(false),
    require_neurofilter_(false),
    nf_dim_latent_(-1),
    nf_train_dropout_p_(0.5),
    nf_leaky_relu_negative_slope_(0.1),
    nf_train_is_gpu_(false),
    nf_infer_is_gpu_(false),
    nf_device_id_(0),
    nf_train_nexample_(-1),
    nf_train_batchsize_(-1),
    nf_train_nepoch_(100),
    nf_train_learning_rate_(0.01),
    nf_train_min_lr_(0.0001),
    nf_train_clip_grad_norm_type_(2),
    nf_train_clip_grad_max_norm_(1),
    nf_train_is_mthread_(false),
    nf_collect_nthread_(-1),
    nf_train_nthread_(4) {
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
      ("ground_truth", po::bool_switch(&is_ground_truth_)->default_value(false),
       "Whether to fetch the ground truths, i.e., linear scan without pruning")
      ("require_neurofilter", po::bool_switch(&require_neurofilter_)->default_value(false),
       "Whether to implant neurofilters")
      ("dim_latent", po::value<ID_TYPE>(&nf_dim_latent_),
       "Dimension of neural model latent variables")
      ("dropout_p", po::value<VALUE_TYPE>(&nf_train_dropout_p_)->default_value(0.5),
       "Dropout probability for MLP latent layer")
      ("leaky_relu_negative_slope", po::value<VALUE_TYPE>(&nf_leaky_relu_negative_slope_)->default_value(0.1),
       "Leaky ReLU negative slope for MLP")
      ("nf_train_is_gpu", po::bool_switch(&nf_train_is_gpu_)->default_value(false),
       "Whether to train neurofilters on GPU (other on CPU)")
      ("nf_infer_is_gpu", po::bool_switch(&nf_infer_is_gpu_)->default_value(false),
       "Whether to run neurofilters on GPU (other on CPU)")
      ("device_id", po::value<ID_TYPE>(&nf_device_id_)->default_value(0),
       "GPU device id")
      ("nf_train_nexample", po::value<ID_TYPE>(&nf_train_nexample_),
       "Number of train examples for neurofilters")
      ("nf_train_batchsize", po::value<ID_TYPE>(&nf_train_batchsize_)->default_value(-1),
       "Neurofilter train batch size")
      ("nf_train_nepoch", po::value<ID_TYPE>(&nf_train_nepoch_)->default_value(100),
       "Neurofilter train (maximal) number of epochs")
      ("learning_rate", po::value<VALUE_TYPE>(&nf_train_learning_rate_)->default_value(0.01),
       "Neurofilter train learning rate")
      ("nf_train_min_lr", po::value<VALUE_TYPE>(&nf_train_min_lr_)->default_value(0.0001),
       "Neurofilter train minimal learning rate, for adjusting learning rates")
      ("clip_grad_norm_type", po::value<VALUE_TYPE>(&nf_train_clip_grad_norm_type_)->default_value(2),
       "Gradient clipping norm type")
      ("clip_grad_max_norm", po::value<VALUE_TYPE>(&nf_train_clip_grad_max_norm_)->default_value(1),
       "Gradient clipping max norm")
      ("nf_query_filepath", po::value<std::string>(&nf_query_filepath_),
       "Query file path to train neurofilters")
      ("nf_train_mthread", po::bool_switch(&nf_train_is_mthread_)->default_value(false),
       "Whether to train neurofilters multithreadingly")
      ("nf_collect_nthread", po::value<ID_TYPE>(&nf_collect_nthread_)->default_value(-1),
       "Number of threads to collect neurofilter train set; default nf_train_nthread")
      ("nf_train_nthread", po::value<ID_TYPE>(&nf_train_nthread_)->default_value(4),
       "Number of threads to train neurofilters");

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
    if (nf_dim_latent_ < 0) {
      nf_dim_latent_ = series_length_;
    }

    if (nf_train_nexample_ < 0) {
      std::cout << "Please specify the number of neurofilter train examples by setting --neurofilter_train_nexample"
                << std::endl;
      exit(-1);
    }

    if (nf_train_batchsize_ < 0) {
      nf_train_batchsize_ = nf_train_nexample_;
    }

    if (vm.count("nf_infer_is_gpu") < 1) {
      nf_infer_is_gpu_ = nf_train_is_gpu_;
    }

    if (vm.count("dim_latent") < 1) {
      nf_dim_latent_ = series_length_;
    }

    if (nf_train_is_mthread_) {
      if (nf_collect_nthread_ < 0) {
        nf_collect_nthread_ = nf_train_nthread_;
      }
    }
  }
}

void dstree::Config::log(upcite::Logger &logger) {
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "db_filepath = %s")
        % db_filepath_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "query_filepath = %s")
        % query_filepath_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "is_znormalized = %d")
        % is_znormalized_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "db_size = %d")
        % db_nseries_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "query_size = %d")
        % query_nseries_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "leaf_size = %d")
        % leaf_max_nseries_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "batch_load_nseries = %d")
        % batch_load_nseries_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "default_nbuffer = %d")
        % default_nbuffer_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "on_disk = %d")
        % on_disk_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "index_persist_folderpath = %s")
        % index_persist_folderpath_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "index_persist_file_postfix = %s")
        % index_persist_file_postfix_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "node_nchild = %d")
        % node_nchild_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "vertical_split_nsubsegment = %d")
        % vertical_split_nsubsegment_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "vertical_split_gain_tradeoff_factor = %.3f")
        % vertical_split_gain_tradeoff_factor_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "is_exact_search = %d")
        % is_exact_search_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "search_max_nseries = %d")
        % search_max_nseries_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "search_max_nnode = %d")
        % search_max_nnode_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "n_nearest_neighbor = %d")
        % n_nearest_neighbor_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "is_ground_truth = %d")
        % is_ground_truth_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "require_neurofilter = %d")
        % require_neurofilter_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_dim_latent = %d")
        % nf_dim_latent_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_leaky_relu_negative_slope = %.3f")
        % nf_leaky_relu_negative_slope_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_dropout_p = %.3f")
        % nf_train_dropout_p_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_is_gpu = %d")
        % nf_train_is_gpu_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_infer_is_gpu = %d")
        % nf_infer_is_gpu_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_device_id = %d")
        % nf_device_id_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_nexample = %d")
        % nf_train_nexample_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_batchsize = %d")
        % nf_train_batchsize_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_nepoch = %d")
        % nf_train_nepoch_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_learning_rate = %.3f")
        % nf_train_learning_rate_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_min_lr = %.3f")
        % nf_train_min_lr_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_clip_grad_norm_type = %.3f")
        % nf_train_clip_grad_norm_type_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_clip_grad_max_norm = %.3f")
        % nf_train_clip_grad_max_norm_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_query_filepath = %s")
        % nf_query_filepath_;

  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_mthread = %d")
        % nf_train_is_mthread_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_collect_nthread = %d")
        % nf_collect_nthread_;
  MALAT_LOG(logger.logger, trivial::info) << boost::format(
        "nf_train_nthread = %d")
        % nf_train_nthread_;
}
