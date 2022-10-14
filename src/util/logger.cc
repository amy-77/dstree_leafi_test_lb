//
// Created by Qitong Wang on 2022/10/3.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "logger.h"

#include <ostream>
#include <fstream>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

namespace expr = boost::log::expressions;
namespace sinks = boost::log::sinks;

void my_formatter(logging::record_view const &rec, logging::formatting_ostream &strm) {
  // https://stackoverflow.com/a/60112004
//  strm << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y%m%d_%H:%M:%S.%f") << " ";
  if (auto timestamp = boost::log::extract<boost::posix_time::ptime>("TimeStamp", rec)) {
    std::tm ts = boost::posix_time::to_tm(*timestamp);

    char buf[128];
    if (std::strftime(buf, sizeof(buf), "%y%m%d%H%M%S", &ts) > 0) {
      strm << buf << " ";
    }
  }

  strm << rec[logging::trivial::severity] << " ";

  logging::value_ref<std::string> absolute_path = logging::extract<std::string>("File", rec);
  strm << boost::filesystem::path(absolute_path.get()).filename().string() << ":";
  strm << logging::extract<int>("Line", rec) << " ";

  strm << rec[expr::smessage];
}

upcite::Logger::Logger(std::string log_filepath) {
  typedef sinks::synchronous_sink<sinks::text_ostream_backend> text_sink;
  boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();

  sink->locked_backend()->add_stream(boost::make_shared<std::ofstream>(log_filepath, std::ios_base::app));
  sink->set_formatter(&my_formatter);

  logging::core::get()->add_sink(sink);
  logging::add_common_attributes();

  logger = std::make_unique<src::severity_logger<logging::trivial::severity_level> >();
}
