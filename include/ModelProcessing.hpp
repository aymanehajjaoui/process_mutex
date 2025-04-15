/*CNNProcessing.hpp*/

#pragma once

#include "SystemUtils.hpp"

#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE 

void model_inference(Channel &channel);
void model_inference_mod(Channel &channel);
