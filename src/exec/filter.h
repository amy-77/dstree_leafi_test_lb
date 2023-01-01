//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_FILTER_H
#define DSTREE_FILTER_H

#include <memory>
#include <functional>

#include <torch/torch.h>

#include "global.h"
#include "config.h"
#include "model.h"

namespace upcite {
namespace dstree {

class Filter {
 public:
  explicit Filter(dstree::Config &config,
                  ID_TYPE id,
                  std::reference_wrapper<torch::Tensor> shared_train_queries);
  ~Filter() = default;

  RESPONSE push_example(VALUE_TYPE bsf_distance, VALUE_TYPE nn_distance) {
    bsf_distances_.push_back(bsf_distance);
    nn_distances_.push_back(nn_distance);

    train_size_ += 1;

    return SUCCESS;
  };

  RESPONSE train();

  VALUE_TYPE infer(torch::Tensor &query_series) const;

 private:
  ID_TYPE id_;

  std::reference_wrapper<dstree::Config> config_;

  std::unique_ptr<FilterModel> model_;
  std::unique_ptr<torch::Device> device_; // TODO ref?

  bool is_trained_;
  ID_TYPE train_size_;

  std::reference_wrapper<torch::Tensor> shared_train_queries_;
  std::vector<VALUE_TYPE> bsf_distances_;
  std::vector<VALUE_TYPE> nn_distances_;

  std::vector<VALUE_TYPE> node_lower_bound_distances_;
//  std::vector<VALUE_TYPE> node_upper_bound_distances_;

//  std::unique_ptr<at::cuda::CUDAStream> stream_;
};

}
}

#endif //DSTREE_FILTER_H
