//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <memory>

#include "logger.h"
#include "config.h"
#include "index.h"

namespace dstree = upcite::dstree;

int main(int argc, char *argv[]) {
  std::shared_ptr<dstree::Config> config =  std::make_shared<dstree::Config>(argc, argv);
  std::shared_ptr<upcite::Logger> logger = std::make_shared<upcite::Logger>(config->log_filepath_);
  config->log(logger);

  std::unique_ptr<dstree::Index> index = std::make_unique<dstree::Index>(config, logger);

  index->build();
  index->dump();

  return 0;
}
