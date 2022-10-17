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
    leaf_max_nseries_(-1),
    batch_load_nseries_(-1),
    default_nbuffer_(1024 * 64),
    index_persist_folderpath_("."),
    index_persist_file_postfix_(".bin"),
    node_nchild_(2),
    vertical_split_nsubsegment_(2),
    vertical_split_gain_tradeoff_factor_(2),
    is_exact_search_(false),
    search_max_nseries_(-1) {
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
      ("leaf_size", po::value<ID_TYPE>(&leaf_max_nseries_)->required(),
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
       "Maximal number of series to be checked during query answering ");

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


  if (is_exact_search_) {
    if (search_max_nseries_ < 1) {
      search_max_nseries_ = db_nseries_;
    }
  } else if (search_max_nseries_ > 0) {
    is_exact_search_ = true;
  }
}

void dstree::Config::log(std::shared_ptr<upcite::Logger> &logger) {
  MALAT_LOG(logger->logger, trivial::info) << boost::format("db_filepath = %s") % db_filepath_;
  MALAT_LOG(logger->logger, trivial::info) << boost::format("query_filepath = %s") % query_filepath_;

  MALAT_LOG(logger->logger, trivial::info) << boost::format("is_znormalized = %d") % is_znormalized_;

  MALAT_LOG(logger->logger, trivial::info) << boost::format("db_size = %d") % db_nseries_;
  MALAT_LOG(logger->logger, trivial::info) << boost::format("query_size = %d") % query_nseries_;
  MALAT_LOG(logger->logger, trivial::info) << boost::format("leaf_size = %d") % leaf_max_nseries_;

  MALAT_LOG(logger->logger, trivial::info) << boost::format("batch_load_nseries = %d") % batch_load_nseries_;
  MALAT_LOG(logger->logger, trivial::info) << boost::format("default_nbuffer = %d") % default_nbuffer_;

  MALAT_LOG(logger->logger, trivial::info) << boost::format("on_disk = %d") % on_disk_;
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("index_persist_folderpath = %s") % index_persist_folderpath_;
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("index_persist_file_postfix = %s") % index_persist_file_postfix_;

  MALAT_LOG(logger->logger, trivial::info) << boost::format("node_nchild = %d") % node_nchild_;
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("vertical_split_nsubsegment = %d") % vertical_split_nsubsegment_;
  MALAT_LOG(logger->logger, trivial::info)
    << boost::format("vertical_split_gain_tradeoff_factor = %.3f") % vertical_split_gain_tradeoff_factor_;
}
