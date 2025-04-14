/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_1_bias[32] = {-327, -464, -326, -110, -205, -227, -224, -360, -155, -255, -191, -280, -110, -232, -239, -62, -351, -345, -287, -252, -445, -228, -218, -343, -180, -284, -37, -164, -376, -338, -330, -172}
;
const int16_t batch_normalization_1_kernel[32] = {348, 878, 906, 766, 307, 860, 282, 419, 823, 862, 687, 538, 490, 539, 613, 492, 944, 281, 580, 1197, 476, 575, 463, 1854, 530, 825, 286, 484, 485, 913, 583, 567}
;