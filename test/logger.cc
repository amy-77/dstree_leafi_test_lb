//
// Created by Qitong Wang on 2022/10/3.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "logger.h"

#include <ostream>
#include <memory>

#include <boost/filesystem.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/manipulators/add_value.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/date_time/posix_time/posix_time_types.hpp>

#include "global.h"

namespace expr = boost::log::expressions;
namespace fs = boost::filesystem;

namespace constant = upcite::constant;

void my_formatter(const logging::record_view &rec, logging::formatting_ostream &strm) {
  // https://stackoverflow.com/a/60112004
//  strm << expr::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y%m%d_%H:%M:%S.%f") << " ";
  if (auto timestamp = boost::log::extract<boost::posix_time::ptime>("TimeStamp", rec)) {
    std::tm ts = boost::posix_time::to_tm(*timestamp);

//    char buf[constant::STR_DEFAULT_SIZE];
    char buf[256];
    if (std::strftime(buf, sizeof(buf), "%y%m%d%H%M%S", &ts) > 0) {
      strm << buf << " ";
    }
  }

  strm << rec[logging::trivial::severity] << " ";

#ifdef DEBUG
  logging::value_ref<std::string> absolute_path = logging::extract<std::string>("File", rec);
  strm << fs::change_extension(fs::path(absolute_path.get()).filename(), "").string() << ":";
  strm << logging::extract<int>("Line", rec) << " ";
#endif

  strm << rec[expr::smessage];
}

upcite::Logger::~Logger() {
  sink_->flush();
}

upcite::Logger::Logger(std::string log_filepath) {
  sink_ = boost::make_shared<sinks::synchronous_sink<sinks::text_ostream_backend>>();

  sink_->locked_backend()->add_stream(boost::make_shared<std::ofstream>(log_filepath, std::ios_base::app));

#ifdef DEBUG
  sink_->locked_backend()->auto_flush(true);
#endif

  sink_->set_formatter(&my_formatter);

  logging::core::get()->add_sink(sink_);
  logging::add_common_attributes();

  logger = std::make_unique<src::severity_logger_mt<logging::trivial::severity_level> >();
}