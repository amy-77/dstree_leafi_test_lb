//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "index.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

int main(int argc, char *argv[]) {
  std::unique_ptr<dstree::Config> config = std::make_unique<dstree::Config>(argc, argv);

  std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt(constant::LOGGER_NAME, config->log_filepath_);

#ifdef DEBUG
  logger->set_pattern("%C-%m-%d %H:%M:%S.%e %L %P:%t %v");
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::debug);
#else
  logger->set_pattern("%C-%m-%d %H:%M:%S.%e %L %v");
  logger->set_level(spdlog::level::info);
  logger->flush_on(spdlog::level::err);
#endif

  spdlog::set_default_logger(logger);

  config->log();

  std::unique_ptr<dstree::Index> index = std::make_unique<dstree::Index>(*config);

  RESPONSE status;

  if (config->to_load_index_) {
    status = index->load();
  } else {
    status = index->build();
  }

  if (config->on_disk_ || config->to_dump_index_) {
    status = static_cast<RESPONSE>(status | index->dump());
  }

  if (status == FAILURE) {
    exit(-1);
  }

  index->search();

  return 0;
}
