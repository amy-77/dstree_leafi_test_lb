//
// Created by Qitong Wang on 2022/12/29.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <iostream>

#include <spdlog/spdlog.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "str.h"
#include "scheduler.h"
#include "distance.h"
#include "global.h"
#include "config.h"
#include "logger.h"
#include "model.h"

namespace dstree = upcite::dstree;

int main(int argc, char *argv[]) {
  srand(static_cast <unsigned> (time(0)));

  ID_TYPE series_length = 256, num_series = 1000;
  ID_TYPE num_epoch = 100;

  std::unique_ptr<torch::Device> device_ = std::make_unique<torch::Device>(
      torch::kCUDA, static_cast<c10::DeviceIndex>(1));

  std::vector<VALUE_TYPE> queries;
  for (ID_TYPE i = 0; i < series_length * num_series; ++i) {
    queries.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  auto query_tsr = torch::from_blob(queries.data(),
                                    {num_series, series_length},
                                    torch::TensorOptions().dtype(TORCH_VALUE_TYPE).device(*device_).requires_grad(false));
//  query_tsr = query_tsr.to(*device_);
//  query_tsr.requires_grad_(false);
  std::cout << query_tsr.device() << std::endl;

//  auto tensor_zero_grad = torch::zeros({num_series, series_length},
//                                       torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
//  tensor_zero_grad = tensor_zero_grad.to(*device_);

  std::vector<VALUE_TYPE> nn_distances_;
  for (ID_TYPE i = 0; i < num_series; ++i) {
    nn_distances_.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  auto target_tsr_cpu = torch::from_blob(nn_distances_.data(),
                                     {num_series},
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE).requires_grad(false));
  auto target_tsr = target_tsr_cpu.to(*device_);
//  target_tsr = target_tsr.to(*device_);
//  target_tsr.requires_grad_(false);

  std::unique_ptr<dstree::FilterModel> model_ = std::make_unique<dstree::MLP>(256, 256, 0.9, 0.1);
  model_->to(*device_);
  std::cout << model_->parameters()[0].device() << std::endl;

  torch::optim::SGD optimizer(model_->parameters(), 0.001);
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));

  auto last_query_cpu = query_tsr.index({query_tsr.size(0) - 1}).cpu().contiguous();
  std::cout << -1 << " " << -1 << " "
            << query_tsr.requires_grad() << " "
            << target_tsr.requires_grad() << " "
            << upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)) << std::endl;

  for (ID_TYPE i = 0; i < num_epoch; ++i) {
    model_->train();

//    auto query_tsr_detached = query_tsr.detach().clone();
//    query_tsr_detached.requires_grad_(false);
    optimizer.zero_grad();

    torch::Tensor prediction = model_->forward(query_tsr);
//    torch::Tensor prediction = model_->forward(query_tsr_detached);

    torch::Tensor loss = mse_loss->forward(prediction, target_tsr);

    loss.backward();
//    query_tsr.grad().data().set_(tensor_zero_grad);

    optimizer.step();

    auto last_query_cpu = query_tsr.index({query_tsr.size(0) - 1}).cpu().contiguous();
    std::cout << i << " " << loss.item<VALUE_TYPE>() << " "
              << query_tsr.requires_grad() << " "
              << target_tsr.requires_grad() << " "
              << upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)) << std::endl;
  }

  std::vector<VALUE_TYPE> queries_valid;
  for (ID_TYPE i = 0; i < series_length * num_series; ++i) {
    queries_valid.push_back(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
  }
  auto queries_valid_tsr = torch::from_blob(queries_valid.data(),
                                            {num_series, series_length},
                                            torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  queries_valid_tsr = queries_valid_tsr.to(*device_);
  queries_valid_tsr.requires_grad_(false);

  for (ID_TYPE i = 0; i < num_epoch; ++i) {
    model_->eval();

    torch::Tensor prediction = model_->forward(queries_valid_tsr);
    torch::Tensor loss = mse_loss->forward(prediction, target_tsr);

    auto last_query_cpu = query_tsr.index({query_tsr.size(0) - 1}).cpu().contiguous();
    std::cout << i << " " << -1 << " "
              << queries_valid_tsr.requires_grad() << " "
              << target_tsr.requires_grad() << " "
              << upcite::get_str(last_query_cpu.data_ptr<VALUE_TYPE>(), last_query_cpu.size(0)) << std::endl;
  }

  return 0;
}