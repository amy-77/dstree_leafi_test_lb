//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <memory>
#include <iostream>
#include <chrono>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "index.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

int main(int argc, char *argv[]) {
  std::cout << "[MAIN] Starting dstree application..." << std::endl;
  
  std::cout << "[MAIN] Step 1: Parsing configuration..." << std::endl;
  std::unique_ptr<dstree::Config> config = std::make_unique<dstree::Config>(argc, argv);
  std::cout << "[MAIN] Configuration parsed successfully" << std::endl;

  std::cout << "[MAIN] Step 2: Setting up logger..." << std::endl;
  std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt(constant::LOGGER_NAME, config->log_filepath_);

#ifdef DEBUG
  logger->set_pattern("%C-%m-%d %H:%M:%S.%e %L %P:%t %v");
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::debug);
  std::cout << "[MAIN] Logger set to DEBUG mode" << std::endl;
#else
  logger->set_pattern("%C-%m-%d %H:%M:%S.%e %L %v");
  logger->set_level(spdlog::level::info);
  logger->flush_on(spdlog::level::err);
  std::cout << "[MAIN] Logger set to RELEASE mode" << std::endl;
#endif

  spdlog::set_default_logger(logger);
  std::cout << "[MAIN] Logger initialized successfully" << std::endl;

  std::cout << "[MAIN] Step 3: Logging configuration..." << std::endl;
  config->log();
  std::cout << "[MAIN] Configuration logged" << std::endl;

  std::cout << "[MAIN] Step 4: Creating index..." << std::endl;
  std::unique_ptr<dstree::Index> index = std::make_unique<dstree::Index>(*config);
  std::cout << "[MAIN] Index object created" << std::endl;

  // 从这里开始计时
  auto total_start_time = std::chrono::high_resolution_clock::now();
  
  RESPONSE status;
  std::chrono::milliseconds index_operation_time(0);

  if (config->to_load_index_) {
    std::cout << "[MAIN] Step 5: Loading existing index..." << std::endl;
    auto step_start_time = std::chrono::high_resolution_clock::now();
    status = index->load();
    auto step_end_time = std::chrono::high_resolution_clock::now();
    index_operation_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end_time - step_start_time);
    if (status == SUCCESS) {
      std::cout << "[MAIN] Index loaded successfully (" << index_operation_time.count() << " ms)" << std::endl;
    } else {
      std::cout << "[MAIN] ERROR: Failed to load index! (" << index_operation_time.count() << " ms)" << std::endl;
    }
  } else {
    std::cout << "[MAIN] Step 5: Building new index..." << std::endl;
    auto step_start_time = std::chrono::high_resolution_clock::now();
    status = index->build();
    auto step_end_time = std::chrono::high_resolution_clock::now();
    index_operation_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end_time - step_start_time);
    if (status == SUCCESS) {
      std::cout << "[MAIN] Index built successfully (" << index_operation_time.count() << " ms)" << std::endl;
    } else {
      std::cout << "[MAIN] ERROR: Failed to build index! (" << index_operation_time.count() << " ms)" << std::endl;
    }
  }

  std::chrono::milliseconds dump_time(0);
  if (config->to_dump_index_) {
    std::cout << "[MAIN] Step 6: Dumping index to disk..." << std::endl;
    auto step_start_time = std::chrono::high_resolution_clock::now();
    RESPONSE dump_status = index->dump();
    auto step_end_time = std::chrono::high_resolution_clock::now();
    dump_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end_time - step_start_time);
    status = static_cast<RESPONSE>(status | dump_status);
    if (dump_status == SUCCESS) {
      std::cout << "[MAIN] Index dumped successfully (" << dump_time.count() << " ms)" << std::endl;
    } else {
      std::cout << "[MAIN] ERROR: Failed to dump index! (" << dump_time.count() << " ms)" << std::endl;
    }
  }

  if (status == FAILURE) {
    std::cout << "[MAIN] CRITICAL ERROR: Previous steps failed, exiting..." << std::endl;
    exit(-1);
  }
 
  std::cout << "[MAIN] Step 7: Starting search..." << std::endl;
  auto step_start_time = std::chrono::high_resolution_clock::now();
  index->search(config->to_profile_search_);
  auto step_end_time = std::chrono::high_resolution_clock::now();
  auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(step_end_time - step_start_time);
  std::cout << "[MAIN] Search completed (" << search_time.count() << " ms)" << std::endl;

  auto total_end_time = std::chrono::high_resolution_clock::now();
  auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end_time - total_start_time);

  std::cout << "\n[MAIN] ===== TIMING SUMMARY =====" << std::endl;
  std::cout << "[MAIN] Index build/load: " << index_operation_time.count() << " ms" << std::endl;
  if (config->to_dump_index_) {
    std::cout << "[MAIN] Index dump: " << dump_time.count() << " ms" << std::endl;
  }
  std::cout << "[MAIN] Search execution: " << search_time.count() << " ms" << std::endl;
  std::cout << "[MAIN] Total core execution time: " << total_time.count() << " ms (" << total_time.count()/1000.0 << " seconds)" << std::endl;
  std::cout << "[MAIN] ============================" << std::endl;

  std::cout << "[MAIN] Program finished successfully!" << std::endl;
  return 0;
}
