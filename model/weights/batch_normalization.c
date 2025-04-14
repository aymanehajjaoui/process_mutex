/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_bias[16] = {-155, -252, -178, -276, -81, -271, -95, -322, -131, -216, -193, -138, -138, -217, -189, -38}
;
const int16_t batch_normalization_kernel[16] = {313, 149, 471, 390, 557, 429, 377, 636, 172, 337, 258, 521, 436, 450, 178, 435}
;