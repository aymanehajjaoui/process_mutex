/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

 // InputLayer is excluded
#include "conv1d.h" // InputLayer is excluded
#include "batch_normalization.h" // InputLayer is excluded
#include "max_pooling1d.h" // InputLayer is excluded
#include "conv1d_1.h" // InputLayer is excluded
#include "batch_normalization_1.h" // InputLayer is excluded
#include "max_pooling1d_1.h" // InputLayer is excluded
#include "conv1d_2.h" // InputLayer is excluded
#include "batch_normalization_2.h" // InputLayer is excluded
#include "max_pooling1d_2.h" // InputLayer is excluded
#include "conv1d_3.h" // InputLayer is excluded
#include "batch_normalization_3.h" // InputLayer is excluded
#include "max_pooling1d_3.h" // InputLayer is excluded
#include "flatten.h" // InputLayer is excluded
#include "dense.h" // InputLayer is excluded
#include "dense_1.h"
#endif


#define MODEL_INPUT_DIM_0 48
#define MODEL_INPUT_DIM_1 1
#define MODEL_INPUT_DIMS 48 * 1

#define MODEL_OUTPUT_SAMPLES 1

#define MODEL_INPUT_SCALE_FACTOR 9 // scale factor of InputLayer
#define MODEL_INPUT_NUMBER_T int16_t
#define MODEL_INPUT_LONG_NUMBER_T int32_t

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef  input_t[48][1];
typedef int16_t input_t[48][1];
typedef dense_1_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif