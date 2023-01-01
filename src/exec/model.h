//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_MODEL_H_
#define DSTREE_SRC_EXEC_MODEL_H_

#include <cmath>

#include <torch/torch.h>

#include "global.h"

namespace upcite {
namespace dstree {

class FilterModel : public torch::nn::Module {
 public:
  virtual ~FilterModel() {};

  virtual torch::Tensor forward(torch::Tensor &x) = 0;
//  virtual torch::Tensor infer(torch::Tensor &x) = 0;

// private:
//  torch::nn::Linear fc1_{nullptr}, fc3_{nullptr};
//  torch::nn::LeakyReLU activate_{nullptr};
};

class MLP : public FilterModel {
 public:
  MLP(ID_TYPE dim_input, ID_TYPE dim_latent, VALUE_TYPE dropout_p, VALUE_TYPE negative_slope) :
      dropout_p_(dropout_p),
      negative_slope_(negative_slope) {
    fc1_ = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
//    fc2 = register_module("fc2", torch::nn::Linear(dim_latent, dim_latent));
    fc3_ = register_module("fc3", torch::nn::Linear(dim_latent, 1));

//    torch::nn::init::kaiming_uniform_(fc1->weight, negative_slope_, torch::kFanIn, torch::kLeakyReLU);
//    torch::nn::init::kaiming_uniform_(fc2->weight, negative_slope_, torch::kFanIn, torch::kLeakyReLU);
//    torch::nn::init::kaiming_uniform_(fc1->weight, 0, torch::kFanIn, torch::kTanh);
//    torch::nn::init::kaiming_uniform_(fc2->weight, 0, torch::kFanIn, torch::kTanh);
//    torch::nn::init::kaiming_uniform_(fc1->weight, 0, torch::kFanIn, torch::kSigmoid);
//    torch::nn::init::kaiming_uniform_(fc2->weight, 0, torch::kFanIn, torch::kSigmoid);
//    torch::nn::init::xavier_uniform_(fc3->weight, 1.0);

    activate_ = register_module("lkrelu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(negative_slope).inplace(true)));
//     activate_ = register_module("sftp", torch::nn::Softplus(torch::nn::SoftplusOptions().beta(0.24).threshold(42.42)));
  }

  torch::Tensor forward(torch::Tensor &x) {
    x = activate_->forward(fc1_->forward(x));
//    x = torch::dropout(x, dropout_p_, is_training());

//    x = activate_->forward(fc2->forward(x));
//    x = torch::dropout(x, dropout_p_, is_training());

    return at::squeeze(fc3_->forward(x));
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
