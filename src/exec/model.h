//
// Created by Qitong Wang on 2022/10/24.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_MODEL_H_
#define DSTREE_SRC_EXEC_MODEL_H_

#include <torch/torch.h>

#include "global.h"

namespace upcite {
namespace dstree {

class NFModel : public torch::nn::Module {
 public:
  virtual ~NFModel() {};

  virtual torch::Tensor forward(torch::Tensor &x) = 0;
  virtual torch::Tensor infer(torch::Tensor &x) = 0;
};

class MLP : public NFModel {
 public:
  MLP(ID_TYPE dim_input, ID_TYPE dim_latent, VALUE_TYPE dropout_p, VALUE_TYPE negative_slope) :
      dropout_p(dropout_p),
      negative_slope(negative_slope) {
    fc1 = register_module("fc1", torch::nn::Linear(dim_input, dim_latent));
//    fc2 = register_module("fc2", torch::nn::Linear(dim_latent, dim_latent));
    fc3 = register_module("fc3", torch::nn::Linear(dim_latent, 1));
  }

  torch::Tensor forward(torch::Tensor &x) {
    x = torch::leaky_relu(fc1->forward(x), negative_slope);
    return at::squeeze(fc3->forward(x));
  }

  torch::Tensor infer(torch::Tensor &x) {
    x = torch::leaky_relu(fc1->forward(x), negative_slope);
    return at::squeeze(fc3->forward(x));
  }

 private:
  VALUE_TYPE dropout_p, negative_slope;

//  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
  torch::nn::Linear fc1{nullptr}, fc3{nullptr};
};

}
}

#endif //DSTREE_SRC_EXEC_MODEL_H_
