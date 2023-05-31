//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_MODEL_H_
#define DSTREE_SRC_EXEC_MODEL_H_

#include <cmath>
#include <vector>
#include <string>

#include <torch/torch.h>
#include "spdlog/spdlog.h"

#include "global.h"
#include "vec.h"
#include "str.h"

namespace upcite {

enum MODEL_TYPE {
  MLP = 0,
  CNN = 1,
  RNN = 2,
  SAN = 3, // self-attention networks, i.e., the Transformer variants
  size = 4
};

static std::vector<MODEL_TYPE> MODEL_TYPE_LIST{
    MLP, CNN, RNN, SAN
};

struct MODEL_SETTING {
 public:
  MODEL_SETTING() {
    model_setting_str = "";

    model_type = MLP;
    num_layer = 2;
    layer_size = 256;
    has_skip_connections = false;

    gpu_mem_mb = -1;

    gpu_ms_per_series = -1;
    cpu_ms_per_series = -1;

    pruning_prob = 0;
  };

  explicit MODEL_SETTING(const std::string& setting_str, std::string delim = "_") {
    model_setting_str = setting_str;

    std::vector<std::string> setting_segments = upcite::split_str(setting_str, delim);

#ifdef DEBUG
    //#ifndef DEBUGGED
    spdlog::debug("model_setting {:d} segments in {:s}",
                  setting_segments.size(), setting_str);
    //#endif
#endif

    // coding-version_model-type_num-layer_dim-layer_skip-connected, e.g., v0_mlp_3_256_f
    if (setting_segments[0][1] == '0') { // version
      if (setting_segments[1] == "mlp") {
        model_type = MLP;

        num_layer = std::stol(setting_segments[2]);
        layer_size = std::stol(setting_segments[3]);
        has_skip_connections = setting_segments[4] == "t";
      } else {
        goto default_branch;  // default
      }
    } else {
      default_branch: // default

      model_type = MLP;
      num_layer = 2;
      layer_size = 256;
      has_skip_connections = false;
    }

    gpu_mem_mb = -1;

    gpu_ms_per_series = -1;
    cpu_ms_per_series = -1;

    pruning_prob = 0;
  };

  ~MODEL_SETTING() = default;

  std::string model_setting_str;

  MODEL_TYPE model_type;
  ID_TYPE num_layer;
  ID_TYPE layer_size;
  bool has_skip_connections;

  VALUE_TYPE gpu_mem_mb;

  VALUE_TYPE gpu_ms_per_series;
  VALUE_TYPE cpu_ms_per_series;

  VALUE_TYPE pruning_prob;
};

extern MODEL_SETTING MODEL_SETTING_PLACEHOLDER;
extern std::reference_wrapper<MODEL_SETTING> MODEL_SETTING_PLACEHOLDER_REF;

namespace dstree {

class FilterCore : public torch::nn::Module {
 public:
  ~FilterCore() override {};

  virtual torch::Tensor forward(torch::Tensor &x) = 0;
};

class MLPFilter : public FilterCore {
 public:
  MLPFilter(ID_TYPE dim_input, ID_TYPE dim_latent, VALUE_TYPE dropout_p, VALUE_TYPE negative_slope) :
      dropout_p_(dropout_p),
      negative_slope_(negative_slope) {
    fc1_ = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
//    fc2 = register_module("fc2", torch::nn::Linear(dim_latent, dim_latent));
    fc3_ = register_module("fc3", torch::nn::Linear(dim_latent, 1));

    activate_ = register_module("lkrelu", torch::nn::LeakyReLU(
        torch::nn::LeakyReLUOptions().negative_slope(negative_slope)));
  }

  torch::Tensor forward(torch::Tensor &x) override {
    auto a1 = fc1_->forward(x);
    auto z1 = activate_->forward(a1);
//    x = torch::dropout(x, dropout_p_, is_training());

//    x = activate_->forward(fc2->forward(x));
//    x = torch::dropout(x, dropout_p_, is_training());

    auto a3 = fc3_->forward(z1);
    return at::squeeze(a3);
  }

 private:
  VALUE_TYPE dropout_p_, negative_slope_;

//  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::nn::Linear fc1_{nullptr}, fc3_{nullptr};

  torch::nn::LeakyReLU activate_{nullptr};
//  torch::nn::Softplus activate_{nullptr};
//  torch::nn::Sigmoid activate_;
//  torch::nn::Tanh activate_;
};

}
}

#endif //DSTREE_SRC_EXEC_MODEL_H_
