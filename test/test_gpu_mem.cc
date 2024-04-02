//
// Created by Qitong Wang on 2024/4/2.
// Copyright (c) 2024 Université Paris Cité. All rights reserved.
//

#include <torch/torch.h>
#include <cuda_runtime_api.h>

int main() {
  // Ensure a CUDA device is available
  if (torch::cuda::is_available()) {
    // Device ID for the GPU you're inquiring about
    int device_id = 0;
    cudaSetDevice(device_id);

    size_t free_memory;
    size_t total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);

    std::cout << "Free memory: " << free_memory / 1024.0 / 1024.0 << " MB\n";
    std::cout << "Total memory: " << total_memory / 1024.0 / 1024.0 << " MB\n";
  } else {
    std::cout << "CUDA is not available." << std::endl;
  }

  return 0;
}
