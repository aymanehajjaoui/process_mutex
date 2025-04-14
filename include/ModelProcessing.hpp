/*CNNProcessing.hpp*/

#pragma once

#include "Common.hpp"
#include "SystemUtils.hpp"

// Enable CMSIS-NN and ARM DSP optimizations
#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE 

// Perform CNN calculations on a given channel
void model_inference(Channel &channel);
void model_inference_mod(Channel &channel);
