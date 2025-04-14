/*CNNProcessing.hpp*/

#pragma once

#include "Common.hpp"        // Defines Channel, flags, data structures
#include "SystemUtils.hpp"   // Utilities like thread priority or CPU affinity

// Enable CMSIS-NN and ARM DSP optimizations for embedded inference
#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE 

// Perform CNN inference on a given channel's data queue
void model_inference(Channel &channel);      // Standard inference
void model_inference_mod(Channel &channel);  // Inference with normalization
