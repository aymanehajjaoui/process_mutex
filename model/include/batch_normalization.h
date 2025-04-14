/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_H_
#define _BATCH_NORMALIZATION_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       46

typedef int16_t batch_normalization_output_type[46][16];

#if 0
void batch_normalization(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_H_