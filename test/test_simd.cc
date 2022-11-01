//
// Created by Qitong Wang on 2022/11/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <chrono>
#include <fstream>
#include <boost/format.hpp>

#include "distance.h"
#include "global.h"
#include "config.h"
#include "logger.h"

namespace dstree = upcite::dstree;

int main(int argc, char *argv[]) {
  std::shared_ptr<dstree::Config> config = std::make_shared<dstree::Config>(argc, argv);
  std::shared_ptr<upcite::Logger> logger = std::make_shared<upcite::Logger>(config->log_filepath_);
  config->log(logger);

  ID_TYPE nseries = 500000;
  ID_TYPE nbytes = sizeof(VALUE_TYPE) * config->series_length_ * nseries * 2;

  auto *series_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), nbytes));

  std::ifstream db_fin(config->db_filepath_, std::ios::in | std::ios::binary);
  db_fin.read(reinterpret_cast<char *>(series_cache), nbytes);
  db_fin.close();

  auto *results_simd = static_cast<VALUE_TYPE *>(std::malloc(sizeof(VALUE_TYPE) * nseries));
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256),
                                                                   sizeof(VALUE_TYPE) * 8));

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (ID_TYPE i = 0; i < nseries; ++i) {
    results_simd[i] = upcite::cal_EDsquare_SIMD_8(series_cache + config->series_length_ * i,
                                                  series_cache + config->series_length_ * (nseries + i),
                                                  config->series_length_,
                                                  m256_fetch_cache);
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  MALAT_LOG(logger->logger, trivial::info) << boost::format(
        "SIMD = %.3fs")
        % (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0);

  auto *results_direct = static_cast<VALUE_TYPE *>(std::malloc(sizeof(VALUE_TYPE) * nseries));

  begin = std::chrono::steady_clock::now();
  for (ID_TYPE i = 0; i < nseries; ++i) {
    results_direct[i] = upcite::cal_EDsquare(series_cache + config->series_length_ * i,
                                             series_cache + config->series_length_ * (nseries + i),
                                             config->series_length_);
  }
  end = std::chrono::steady_clock::now();

  MALAT_LOG(logger->logger, trivial::info) << boost::format(
        "direct = %.3fs")
        % (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000000.0);

  for (ID_TYPE i = 0; i < nseries; ++i) {
    if (abs(results_direct[i] - results_simd[i]) > 1e-3) {
      MALAT_LOG(logger->logger, trivial::error) << boost::format(
            "SIMD = %.3f, direct = %.3f")
            % results_simd[i]
            % results_direct[i];
    }
  }

  std::free(series_cache);
  std::free(results_simd);
  std::free(m256_fetch_cache);
  std::free(results_direct);

  return SUCCESS;
}
