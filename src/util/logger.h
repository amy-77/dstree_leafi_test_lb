//
// Created by Qitong Wang on 2022/10/3.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_UTIL_LOGGER_H_
#define DSTREE_SRC_UTIL_LOGGER_H_

#include <string>
#include <memory>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>

namespace logging = boost::log;
namespace trivial = boost::log::trivial;
namespace src = boost::log::sources;

// cr: https://stackoverflow.com/a/31160870
#define MALAT_LOG(log_, sv) BOOST_LOG_SEV((*log_), (sv)) \
  << logging::add_value("Line", (__LINE__)) \
  << logging::add_value("File", (__FILE__)) \
  << logging::add_value("Function", (BOOST_CURRENT_FUNCTION))

namespace upcite {

class Logger {
 public:
  explicit Logger(std::string log_filepath);
  ~Logger() = default;

  std::unique_ptr<src::severity_logger<logging::trivial::severity_level> > logger;
};

}
#endif //DSTREE_SRC_UTIL_LOGGER_H_
