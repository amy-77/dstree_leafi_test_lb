//
// Created by Qitong Wang on 2023/10/31.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include <iostream>
#include <chrono>
#include <memory>

#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional/linear.h>

namespace F = torch::nn::functional;

//#include "filter_core.h"

struct MLP : public torch::nn::Module {
  MLP(int dim_input, int dim_latent) {
    fc1_ = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
    activate_ = register_module("lk_relu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(0.1)));

    fc2_ = register_module("fc2", torch::nn::Linear(dim_latent, 1));
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

  auto device = torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(1));

  int series_length = 256, num_series = 1;

  auto series_arr = static_cast<float *>(malloc(sizeof(float) * series_length * num_series));
  for (int i = 0; i < series_length * num_series; ++i) {
    series_arr[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  auto series_tensor = torch::from_blob(
      series_arr, {num_series, series_length},
      torch::TensorOptions().dtype(torch::kFloat32)).to(device);

  int num_models = 1000;
  std::vector<std::shared_ptr<MLP>> models;

  for (int i = 0; i < num_models; ++i) {
    models.emplace_back(std::make_shared<MLP>(series_length, series_length));
    models[i]->to(device);
    models[i]->eval();
  }
  c10::InferenceMode guard;

  std::vector<float> predictions;
  predictions.reserve(num_models);

  int num_epoch = 1;
  // int num_epoch = 20;

  // without vmap
  auto start = std::chrono::high_resolution_clock::now();
  for (int epoch_i = 0; epoch_i < num_epoch; ++epoch_i) {
    predictions.clear();
    for (int model_i = 0; model_i < num_models; ++model_i) {
      predictions.push_back(models[model_i]->forward(series_tensor).item<float>());
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "one by one = " << duration.count() << "ms" << std::endl;

  // create vmap
  std::vector<torch::Tensor> parameters_l1_a;
  for (int i = 0; i < num_models; ++i) {
    parameters_l1_a.emplace_back(models[i]->parameters()[0]);
  }
  torch::Tensor parameters_l1_a_vmap = torch::stack(parameters_l1_a);
  std::cout << parameters_l1_a_vmap.sizes() << std::endl;

  std::vector<torch::Tensor> parameters_l1_b;
  for (int i = 0; i < num_models; ++i) {
    parameters_l1_b.emplace_back(models[i]->parameters()[1]);
  }
  torch::Tensor parameters_l1_b_vmap = torch::stack(parameters_l1_b);
  std::cout << parameters_l1_b_vmap.sizes() << std::endl;

  std::vector<torch::Tensor> parameters_l2_a;
  for (int i = 0; i < num_models; ++i) {
    parameters_l2_a.emplace_back(models[i]->parameters()[2]);
  }
  torch::Tensor parameters_l2_a_vmap = torch::stack(parameters_l2_a);
  std::cout << parameters_l2_a_vmap.sizes() << std::endl;

  std::vector<torch::Tensor> parameters_l2_b;
  for (int i = 0; i < num_models; ++i) {
    parameters_l2_b.emplace_back(models[i]->parameters()[3]);
  }
  torch::Tensor parameters_l2_b_vmap = torch::stack(parameters_l2_b);
  std::cout << parameters_l2_b_vmap.sizes() << std::endl;

  std::cout << series_tensor.sizes() << std::endl;

  torch::Tensor a2_cpu;
  auto input = series_tensor.view({1, 1, 256});
  // with vmap
  start = std::chrono::high_resolution_clock::now();
  for (int epoch_i = 0; epoch_i < num_epoch; ++epoch_i) {
    std::cout << "input " << input.sizes() << std::endl;
    auto a1 = input.matmul(parameters_l1_a_vmap.transpose(1, 2)) + parameters_l1_b_vmap;
    std::cout << "a1 " << a1.sizes() << std::endl;

    auto z1 = F::leaky_relu(a1, F::LeakyReLUFuncOptions().negative_slope(0.01));
    std::cout << "z1 " << z1.sizes() << std::endl;

    auto a2 = z1.matmul(parameters_l2_a_vmap.transpose(1, 2)) + parameters_l2_b_vmap;
    std::cout << "a2 " << a2.sizes() << std::endl;
    a2_cpu = a2.cpu();
  }
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::cout << "vmap = " << duration.count() << "ms" << std::endl;

  // std::cout << predictions.size() << " " << a2_cpu.sizes() << std::endl;
  // std::cout << predictions[0] << " " << a2_cpu[0] << std::endl;

  free(series_arr);

  return 0;
}
