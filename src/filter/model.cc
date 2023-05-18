//
// Created by Qitong Wang on 2023/5/16.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "model.h"

upcite::MODEL_SETTING upcite::MODEL_SETTING_PLACEHOLDER = upcite::MODEL_SETTING();
std::reference_wrapper<upcite::MODEL_SETTING>
    upcite::MODEL_SETTING_PLACEHOLDER_REF = std::ref(upcite::MODEL_SETTING_PLACEHOLDER);
