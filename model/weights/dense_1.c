/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 16
#define FC_UNITS 1


const int16_t dense_1_bias[FC_UNITS] = {4}
;

const int16_t dense_1_kernel[FC_UNITS][INPUT_SAMPLES] = {{-22, -29, -50, 30, 34, 33, 23, -16, -19, 10, -23, -22, -37, 35, 33, -47}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS