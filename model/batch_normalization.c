/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       46
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR);
      tmp += (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[x][z] = clamp_to(NUMBER_T, tmp);
      }
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR