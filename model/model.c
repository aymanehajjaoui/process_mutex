/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>

 // InputLayer is excluded
#include "conv1d.c"
#include "weights/conv1d.c" // InputLayer is excluded
#include "batch_normalization.c"
#include "weights/batch_normalization.c" // InputLayer is excluded
#include "max_pooling1d.c" // InputLayer is excluded
#include "conv1d_1.c"
#include "weights/conv1d_1.c" // InputLayer is excluded
#include "batch_normalization_1.c"
#include "weights/batch_normalization_1.c" // InputLayer is excluded
#include "max_pooling1d_1.c" // InputLayer is excluded
#include "conv1d_2.c"
#include "weights/conv1d_2.c" // InputLayer is excluded
#include "batch_normalization_2.c"
#include "weights/batch_normalization_2.c" // InputLayer is excluded
#include "max_pooling1d_2.c" // InputLayer is excluded
#include "conv1d_3.c"
#include "weights/conv1d_3.c" // InputLayer is excluded
#include "batch_normalization_3.c"
#include "weights/batch_normalization_3.c" // InputLayer is excluded
#include "max_pooling1d_3.c" // InputLayer is excluded
#include "flatten.c" // InputLayer is excluded
#include "dense.c"
#include "weights/dense.c" // InputLayer is excluded
#include "dense_1.c"
#include "weights/dense_1.c"
#endif


void cnn(
  const input_t input,
  dense_1_output_type dense_1_output) {
  
  // Output array allocation
  static union {
    conv1d_output_type conv1d_output;
    max_pooling1d_output_type max_pooling1d_output;
    batch_normalization_1_output_type batch_normalization_1_output;
    conv1d_2_output_type conv1d_2_output;
    max_pooling1d_2_output_type max_pooling1d_2_output;
    batch_normalization_3_output_type batch_normalization_3_output;
    dense_output_type dense_output;
  } activations1;

  static union {
    batch_normalization_output_type batch_normalization_output;
    conv1d_1_output_type conv1d_1_output;
    max_pooling1d_1_output_type max_pooling1d_1_output;
    batch_normalization_2_output_type batch_normalization_2_output;
    conv1d_3_output_type conv1d_3_output;
    max_pooling1d_3_output_type max_pooling1d_3_output;
    flatten_output_type flatten_output;
  } activations2;


// Model layers call chain 
  
  
  conv1d( // First layer uses input passed as model parameter
    input,
    conv1d_kernel,
    conv1d_bias,
    activations1.conv1d_output
    );
  
  
  batch_normalization(
    activations1.conv1d_output,
    batch_normalization_kernel,
    batch_normalization_bias,
    activations2.batch_normalization_output
    );
  
  
  max_pooling1d(
    activations2.batch_normalization_output,
    activations1.max_pooling1d_output
    );
  
  
  conv1d_1(
    activations1.max_pooling1d_output,
    conv1d_1_kernel,
    conv1d_1_bias,
    activations2.conv1d_1_output
    );
  
  
  batch_normalization_1(
    activations2.conv1d_1_output,
    batch_normalization_1_kernel,
    batch_normalization_1_bias,
    activations1.batch_normalization_1_output
    );
  
  
  max_pooling1d_1(
    activations1.batch_normalization_1_output,
    activations2.max_pooling1d_1_output
    );
  
  
  conv1d_2(
    activations2.max_pooling1d_1_output,
    conv1d_2_kernel,
    conv1d_2_bias,
    activations1.conv1d_2_output
    );
  
  
  batch_normalization_2(
    activations1.conv1d_2_output,
    batch_normalization_2_kernel,
    batch_normalization_2_bias,
    activations2.batch_normalization_2_output
    );
  
  
  max_pooling1d_2(
    activations2.batch_normalization_2_output,
    activations1.max_pooling1d_2_output
    );
  
  
  conv1d_3(
    activations1.max_pooling1d_2_output,
    conv1d_3_kernel,
    conv1d_3_bias,
    activations2.conv1d_3_output
    );
  
  
  batch_normalization_3(
    activations2.conv1d_3_output,
    batch_normalization_3_kernel,
    batch_normalization_3_bias,
    activations1.batch_normalization_3_output
    );
  
  
  max_pooling1d_3(
    activations1.batch_normalization_3_output,
    activations2.max_pooling1d_3_output
    );
  
  
  flatten(
    activations2.max_pooling1d_3_output,
    activations2.flatten_output
    );
  
  
  dense(
    activations2.flatten_output,
    dense_kernel,
    dense_bias,
    activations1.dense_output
    );
  
  
  dense_1(
    activations1.dense_output,
    dense_1_kernel,
    dense_1_bias,// Last layer uses output passed as model parameter
    dense_1_output
    );
}

#ifdef __cplusplus
} // extern "C"
#endif