//
// Created by Qitong Wang on 2023/10/31.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <memory>

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

//#include "filter_core.h"

struct MLP : public torch::nn::Module {
  MLP(int dim_input, int dim_latent) {
    fc1_ = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
    fc2_ = register_module("fc2", torch::nn::Linear(dim_latent, 1));

    activate_ = register_module("lk_relu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.1)));
  }

  torch::Tensor forward(torch::Tensor &x) {
    auto a1 = fc1_->forward(x);
    auto z1 = activate_->forward(a1);

    auto a2 = fc2_->forward(z1);
    return at::squeeze(a2);
  }

  torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
  torch::nn::LeakyReLU activate_{nullptr};
};


int main(int argc, char *argv[]) {
  srand(static_cast <unsigned> (time(nullptr)));

  int series_length = 256, num_series = 1000;
  int num_epoch = 1000;

  auto series_arr = static_cast<float *>(malloc(sizeof(float) * series_length * num_series));
  for (int i = 0; i < series_length * num_series; ++i) {
    series_arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  auto parameters_arr = static_cast<float *>(malloc(sizeof(float) * series_length * (series_length + 1)));
  for (int i = 0; i < series_length * (series_length + 1); ++i) {
    series_arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }

  std::vector<float> predictions;
  predictions.reserve(num_series);

  int k, l;
  int series_offset, parameter_offset;
  float temp_result;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_epoch; ++i) {
    for (int j = 0; j < num_series; ++j) {
      series_offset = series_length * j;
      temp_result = 0;

      for (k = 0; k < series_length + 1; ++k) {
        parameter_offset = series_length * k;

        for (l = 0; l < series_length; ++l) {
          temp_result += series_arr[series_offset + l] * parameters_arr[parameter_offset + l];
        }
      }

      predictions.push_back(temp_result);
    }
    predictions.clear();
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "cpu raw = " << duration.count() << "ms" << std::endl;

  std::vector<torch::Tensor> series_tensor_cpu_vec;
  for (int i = 0; i < num_series; ++i) {
    series_tensor_cpu_vec.emplace_back(torch::from_blob(
        series_arr + series_length * i,
        {1, series_length},
        torch::TensorOptions().dtype(torch::kFloat32)));
  }
  std::cout << "tensor_cpu.device = " << series_tensor_cpu_vec[0].device() << std::endl;

  auto device_cpu = torch::Device(torch::kCPU);
//  auto model = std::make_shared<upcite::dstree::MLPFilter>(series_length, series_length, 0.5, 0.01);
  auto model = std::make_shared<MLP>(series_length, series_length);
  model->to(device_cpu);
  model->eval();
  std::cout << "model->device = " << model->parameters()[0].device() << std::endl;

  c10::InferenceMode guard;

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_epoch; ++i) {
    for (int j = 0; j < num_series; ++j) {
      auto pred = model->forward(series_tensor_cpu_vec[j]);
      predictions.push_back(pred.item<float>());
    }
    predictions.clear();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "cpu libtorch = " << duration.count() << "ms" << std::endl;

  auto device_cuda = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(1));
  std::vector<torch::Tensor> series_tensor_gpu_vec;
  for (int i = 0; i < num_series; ++i) {
    series_tensor_gpu_vec.emplace_back(series_tensor_cpu_vec[i].to(device_cuda));
  }
  std::cout << "tensor_gpu.device = " << series_tensor_gpu_vec[0].device() << std::endl;

  model->to(device_cuda);
  std::cout << "model->device = " << model->parameters()[0].device() << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_epoch; ++i) {
    for (int j = 0; j < num_series; ++j) {
      auto pred = model->forward(series_tensor_gpu_vec[j]);
      predictions.push_back(pred.item<float>());
    }
    predictions.clear();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "gpu libtorch = " << duration.count() << "ms" << std::endl;

  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_epoch; ++i) {
    for (int j = 0; j < num_series; ++j) {
      auto pred = model->forward(series_tensor_gpu_vec[j]);
    }
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "gpu cal_only = " << duration.count() << "ms" << std::endl;


  free(series_arr);
  return 0;
}