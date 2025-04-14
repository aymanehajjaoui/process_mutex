#define SINGLE_FILE
/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define True 1
#define False 0

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor) scale_number_t_ ## type (number, scale_factor)
#define scale(type, number, scale_factor) _scale(type, number, scale_factor)

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN		// Max value for this numeric type
// #define NUMBER_MAX		// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef  number_t;		// Standard size numeric type used for weights and activations
// typedef  long_number_t;	// Long numeric type used for intermediate results

#define NUMBER_MIN_INT16_T -32768
#define NUMBER_MAX_INT16_T 32767

static inline int32_t min_int16_t(
    int32_t a,
    int32_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int32_t max_int16_t(
    int32_t a,
    int32_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int32_t scale_number_t_int16_t(
  int32_t number, int scale_factor) {
  if (scale_factor < 0)
    return number << - scale_factor;
  else 
    return number >> scale_factor;
}
static inline int16_t clamp_to_number_t_int16_t(
  int32_t number) {
	return (int16_t) max_int16_t(
      NUMBER_MIN_INT16_T,
      min_int16_t(
        NUMBER_MAX_INT16_T, number));
}

#define NUMBER_MIN_INT32_T -2147483648
#define NUMBER_MAX_INT32_T 2147483647

static inline int64_t min_int32_t(
    int64_t a,
    int64_t b) {
	if (a <= b)
		return a;
	return b;
}

static inline int64_t max_int32_t(
    int64_t a,
    int64_t b) {
	if (a >= b)
		return a;
	return b;
}

static inline int64_t scale_number_t_int32_t(
  int64_t number, int scale_factor) {
  if (scale_factor < 0)
    return number << - scale_factor;
  else 
    return number >> scale_factor;
}
static inline int32_t clamp_to_number_t_int32_t(
  int64_t number) {
	return (int32_t) max_int32_t(
      NUMBER_MIN_INT32_T,
      min_int32_t(
        NUMBER_MAX_INT32_T, number));
}




static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_H_
#define _CONV1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       48
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      1
#define INPUT_SAMPLES       48
#define CONV_FILTERS        16
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else


  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    1
#define CONV_FILTERS      16
#define CONV_KERNEL_SIZE  3


const int16_t  conv1d_bias[CONV_FILTERS] = {-49, -96, -19, 270, -63, 138, 157, 100, -160, 86, 302, -68, 20, 168, 182, 9}
;

const int16_t  conv1d_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{68}
, {-10}
, {161}
}
, {{-167}
, {-109}
, {-146}
}
, {{-72}
, {103}
, {60}
}
, {{104}
, {66}
, {-47}
}
, {{56}
, {127}
, {-111}
}
, {{-167}
, {151}
, {-26}
}
, {{152}
, {-83}
, {51}
}
, {{-6}
, {137}
, {-189}
}
, {{178}
, {127}
, {72}
}
, {{-86}
, {-106}
, {16}
}
, {{94}
, {102}
, {32}
}
, {{-147}
, {43}
, {165}
}
, {{215}
, {-2}
, {-158}
}
, {{-124}
, {-86}
, {157}
}
, {{145}
, {85}
, {227}
}
, {{62}
, {-184}
, {151}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
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
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_H_
#define _MAX_POOLING1D_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   46
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  16
#define INPUT_SAMPLES   46
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_1_H_
#define _CONV1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       23
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_1_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_1_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      16
#define INPUT_SAMPLES       23
#define CONV_FILTERS        32
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else


  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    16
#define CONV_FILTERS      32
#define CONV_KERNEL_SIZE  3


const int16_t  conv1d_1_bias[CONV_FILTERS] = {83, 64, 19, -110, 35, -83, 77, -12, -38, -59, -7, -44, 8, 21, 40, -15, -72, 173, 17, -165, 188, -71, 15, -104, 45, -67, 55, -40, 2, -108, -58, 18}
;

const int16_t  conv1d_1_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{19, -106, -29, 38, -42, 19, 84, 26, -5, -61, 64, 18, 73, 64, -101, -1}
, {11, -94, -47, 72, 35, -40, 84, 122, 70, -235, -4, 3, -9, -172, 79, -143}
, {-159, 162, -100, 32, 137, -98, -52, 223, -124, 63, -33, 201, 284, 102, -124, -47}
}
, {{-52, -153, 117, -46, -80, -24, 2, -156, 35, -154, 77, -134, -96, -279, -24, -159}
, {21, -199, -137, -184, -187, -89, -39, 139, 7, -174, -98, -191, -71, -22, 81, 164}
, {120, -47, 84, -55, -62, 81, 245, 103, 8, -146, -30, 118, 27, 11, 77, 111}
}
, {{66, -26, -41, 1, 10, -46, -121, 41, 115, 38, 88, 147, 72, 25, 19, 30}
, {131, -29, 9, -24, 117, 60, -82, 52, -110, 157, -25, -203, 66, 187, 44, 107}
, {192, -59, -252, -209, -172, -5, 28, -400, -92, -167, -182, -245, -300, -20, -22, 6}
}
, {{-180, -27, -10, -138, -126, -57, -232, 42, -76, 5, -78, 10, -176, 26, -117, 3}
, {72, 53, 143, 15, -125, -10, -8, -187, 155, -63, 145, 102, -70, -30, 136, -80}
, {114, -12, -23, -41, 37, -116, -113, -139, -61, 67, -123, 67, 23, 157, 27, 123}
}
, {{-30, -26, -7, -100, 93, 103, -106, 4, -137, 141, -121, 197, 59, 188, -134, 3}
, {152, -70, 88, 11, 9, 51, 33, -96, -81, -33, 20, 76, -108, 149, 122, 91}
, {61, -5, 44, 93, 4, -175, -8, 18, -93, -98, -24, 106, 1, -176, -68, 54}
}
, {{-83, -72, -33, 18, -58, 71, -72, 125, -268, -209, -106, 142, -47, -37, -2, 90}
, {105, -108, -54, 57, 41, 54, 91, -25, -28, -75, -72, -81, -3, -117, 13, 84}
, {-163, 158, 22, -117, -51, -39, -166, -67, -42, 290, -183, 7, 23, -88, -114, 130}
}
, {{93, -137, -8, 127, 42, 165, 58, 137, 104, -6, 17, -13, 89, 49, -7, 158}
, {131, 64, -94, -48, -76, 65, 19, -167, 86, -24, 106, 53, -116, -6, 91, -13}
, {19, -227, -29, 50, 11, -279, 19, 31, -16, 29, -56, -87, 78, 82, 160, 52}
}
, {{28, -78, 60, 14, 103, 22, 66, -76, 111, 145, 95, -14, 80, 282, 41, 53}
, {-127, -139, -156, 27, 56, -42, 158, -52, -41, 40, 3, -166, 175, 68, -52, 29}
, {-4, -330, 41, -29, 115, 109, 12, -42, -239, -127, -129, 5, -82, 103, -191, 58}
}
, {{55, -43, 53, -25, 145, 218, 58, -35, -108, -64, 70, 185, 82, 43, 8, 52}
, {13, 135, -75, 74, -105, -232, -37, -131, 4, -40, 64, 31, -155, -455, -19, -506}
, {-115, 70, 211, -46, -5, -172, -73, -69, 84, -15, 22, 62, -71, -87, -88, 253}
}
, {{183, 189, -14, -249, -290, 35, -19, -414, 8, 78, 18, -65, -227, -64, 125, -139}
, {-14, -104, 124, -21, 94, 91, -55, 40, 53, 38, 160, 76, -7, -27, -75, 26}
, {-32, -148, -87, -74, 134, 32, 119, -112, -105, -297, -145, -138, 95, -23, 36, 73}
}
, {{104, -77, 7, -3, 77, 98, 15, 21, 37, -81, 107, 83, 2, 14, 99, -142}
, {-22, -170, -21, 22, -53, 26, 79, 16, -42, 50, -8, 16, 29, 49, -193, 353}
, {19, 74, -130, -67, -168, 155, 156, -81, 45, -206, 69, -98, -236, -161, 156, -237}
}
, {{-183, 53, 20, 99, 95, 116, 45, -55, -26, 108, -49, -33, 41, 246, -175, -97}
, {-12, -35, -54, -177, 51, 161, 3, -104, -54, 179, -150, -133, -4, 91, 63, 11}
, {-164, -168, -13, 175, 112, -11, -60, -13, -157, -141, 80, -32, 150, 50, -127, 1}
}
, {{-84, -77, 76, 94, 55, -155, -15, -94, -50, 36, 53, -108, 175, -167, 114, 73}
, {84, 80, -81, -155, -168, 41, 98, -159, 43, -184, 71, -83, 52, -162, -81, -3}
, {-14, -67, -44, -61, 94, 19, -20, 231, -106, 64, -150, 120, 162, -77, -142, 30}
}
, {{163, 24, -14, 92, 135, -158, 94, 73, -83, -68, 56, -44, 10, 104, 145, 3}
, {52, -46, 16, -99, -33, 4, 59, -72, 173, -92, -19, -130, -152, -336, -46, -277}
, {-308, 49, 43, -157, -297, -103, -239, 266, -106, 187, -199, 85, -123, 5, -170, 17}
}
, {{155, 174, -98, -84, 45, -192, -192, 38, 137, 85, 125, -233, -85, -188, 161, -156}
, {-133, -76, -101, -30, -5, -86, 84, -9, -54, -118, -18, -15, 83, 39, -142, 194}
, {175, 242, 15, -185, -90, -8, 2, 30, 66, 78, -98, 37, -87, 21, 0, -90}
}
, {{-8, -88, -8, 126, 104, -132, 101, -80, -141, -246, 84, 2, 156, -237, -100, -106}
, {-200, -87, 104, -92, 73, -101, -179, 9, -22, 134, -101, 199, 46, 43, -122, -15}
, {-169, 90, 141, -128, -10, -58, 23, -67, -85, 80, -91, 21, 29, 51, -16, -49}
}
, {{59, -118, 37, 161, 103, 123, -83, -19, -63, -201, -70, 34, 83, 78, -23, 128}
, {-86, 51, -99, -142, -176, 28, 111, -582, 1, -104, -32, -137, -147, 58, 13, -101}
, {172, 95, 67, -2, 126, -184, -225, -115, 16, 57, 135, 124, -130, -50, 114, 51}
}
, {{-130, 0, 33, -148, -162, -18, 12, 28, -17, -4, -118, 4, -44, 50, -154, -23}
, {-36, 5, 54, -1, 85, 122, 19, 262, 6, -113, -37, 44, -8, -120, -132, -1}
, {151, 72, 68, 104, -85, -35, 71, -118, 55, 211, 75, 96, -8, 220, 121, 104}
}
, {{-185, 27, -65, -110, 1, 148, -50, -160, -1, 171, -48, 107, 22, 125, -169, 105}
, {-29, -73, -141, -136, -66, 10, 125, 2, -73, -95, -6, -137, -193, -9, 11, 68}
, {29, 9, 76, 43, -181, -33, 18, -68, 14, 27, -91, 10, -9, 96, -148, 175}
}
, {{-57, 14, 28, 122, 90, 60, 62, -67, 68, 69, 27, -138, 202, -98, 11, 91}
, {64, -27, -68, -27, -197, -217, 65, 131, 160, -79, -114, -285, -117, -360, 8, -78}
, {-90, -139, -136, -58, -155, -85, 47, 16, -107, -109, 18, -14, -124, -78, 132, 135}
}
, {{55, 11, -176, -146, 61, -172, -121, 61, -168, 61, -40, -122, 41, 39, 12, 129}
, {-22, 208, -46, -139, -40, -162, -3, -194, -107, 181, -125, -23, 115, 93, 57, 10}
, {48, -79, -32, -41, 2, -21, 17, 21, 65, 29, 81, -71, -75, -22, -2, 2}
}
, {{91, 131, 19, 160, 37, -137, 81, -142, 181, -57, -20, -150, 25, -250, 106, -326}
, {44, 64, -23, -51, 23, -123, -165, 123, 83, 27, 141, -64, -1, 11, -30, 138}
, {-144, -134, 38, 50, 94, 148, 93, 43, -92, -147, -2, -21, 19, 50, 41, -14}
}
, {{61, -21, -1, 120, 210, 133, 5, 39, -227, -40, -64, -8, 195, 92, 61, 84}
, {-137, 9, 64, -105, -131, 19, 73, 94, -100, 81, -138, 27, -78, 23, -203, -101}
, {46, -13, -56, 31, -48, -23, 107, 189, -171, -69, -14, -5, -92, -150, -47, 115}
}
, {{20, -36, -73, -83, -57, 170, 121, 139, -68, -142, -225, -51, -95, -35, -34, -36}
, {48, -90, -19, 120, -62, -5, 35, 0, -93, 4, -13, -141, -73, 3, 63, 121}
, {38, -96, -157, -37, -124, 153, 217, 21, -3, -92, -83, -117, -122, -13, -93, 144}
}
, {{13, -174, 1, -71, 118, 227, -16, 66, -27, 130, -2, 4, 83, 288, -72, 87}
, {63, 10, -31, -58, 36, 89, -44, -89, -32, -36, 115, 1, 57, -37, 61, -198}
, {42, 203, -168, -4, -148, -143, -49, 12, -14, 26, 15, -331, -122, -198, 108, -103}
}
, {{-24, -1, -108, -153, -80, -42, -6, 21, -69, -151, -135, -215, -193, -4, -14, 22}
, {-32, -8, -58, -107, -187, 30, -37, 179, -72, -60, -115, 72, -331, -14, -37, -78}
, {-152, -42, 9, -10, -10, 18, -34, -59, 11, 103, -73, 87, -28, 95, -30, 103}
}
, {{-54, 22, 23, 143, -43, 128, 41, -18, -29, -16, 42, 62, -50, -15, -68, -91}
, {19, 35, 79, 17, 38, 33, 92, 99, -230, 65, -130, 29, -52, 127, -207, 82}
, {19, -255, 113, -118, -166, 60, 4, -166, -5, -41, 78, 206, -60, 207, 147, 170}
}
, {{56, -295, 78, 5, -30, -155, 4, 90, -121, -191, -51, -92, 169, -87, -29, -27}
, {-57, -273, 19, 112, -161, 109, -102, 89, -52, -96, -41, 125, 84, 29, -55, -57}
, {-12, -275, 120, 22, 66, 64, 154, -23, 19, -92, 101, 51, 95, -10, -6, 132}
}
, {{-150, -48, 50, -133, 9, -69, -62, 54, -23, 160, -247, 46, 39, -19, -302, 248}
, {-108, 25, 113, 79, 124, 65, 41, 0, -188, -115, 86, 65, -64, -19, -170, -128}
, {-159, -185, -82, -108, 161, 33, 85, 44, -186, 115, -63, -21, 39, 209, -298, 108}
}
, {{131, 193, -99, 86, 78, 118, 35, -15, 145, 121, 81, -88, 85, 200, 80, 188}
, {52, -59, -133, -190, -180, 49, 8, -99, -41, -82, -221, -101, -152, 0, -47, 51}
, {-92, -124, -89, -78, 3, 164, -132, -47, -95, -135, -119, -156, -211, 48, -253, -9}
}
, {{-213, -96, 118, -137, -15, 140, -43, -48, -89, 31, -64, 1, -99, -44, -116, -52}
, {5, -121, -199, -112, -92, -105, -112, 51, -61, 23, -90, -225, -9, -21, -135, 57}
, {-254, 103, 149, -146, 71, 98, -79, 187, -119, 74, -79, 171, 120, -89, -305, -60}
}
, {{188, -138, 45, -118, -42, 32, 28, 13, -91, -226, -44, -3, -70, -163, 66, 54}
, {-81, 54, -75, -103, -9, -160, -9, -146, -6, -68, -134, 110, 17, 57, -58, 91}
, {-158, -94, 37, -42, -55, 151, -133, 84, -170, 73, 53, 185, -23, 93, -36, 180}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_1_H_
#define _BATCH_NORMALIZATION_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       21

typedef int16_t batch_normalization_1_output_type[21][32];

#if 0
void batch_normalization_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_1_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_1_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_1.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       21
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_1_output_type output) {                // OUT

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
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_1_H_
#define _MAX_POOLING1D_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   21
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_1_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_1(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_1_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  32
#define INPUT_SAMPLES   21
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_1(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_2_H_
#define _CONV1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       10
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_2_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_2_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      32
#define INPUT_SAMPLES       10
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else


  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    32
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t  conv1d_2_bias[CONV_FILTERS] = {14, -91, -76, 88, 64, 101, 13, 25, -55, -56, 26, -32, 57, 72, 41, 22, 3, -19, 32, -52, 19, -32, 43, 76, 103, 73, 47, 26, 84, 86, 1, -104, -8, 52, 52, 14, -10, 54, 60, 32, -92, 102, 110, 25, -21, 59, -2, -73, 76, -2, -81, 3, 63, 75, 98, -64, 2, -67, -22, 66, -82, -7, -28, 47}
;

const int16_t  conv1d_2_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{48, 39, -93, 61, -53, -55, 16, -110, 48, -25, -13, -111, 68, 21, -71, -52, -15, -24, -85, -29, -151, 12, -59, 10, -19, -38, -99, 41, -162, -58, -31, -47}
, {-59, -23, -45, -14, 8, -54, -6, 143, 25, -18, -6, -76, -41, 22, -78, 69, -16, -101, 59, 113, -86, 101, 39, 42, -28, -24, -63, 25, 43, -61, 50, -26}
, {-84, -4, 85, -103, -19, 20, -103, 49, 23, -47, -12, -65, 5, 55, -33, -6, -36, -27, 21, 154, -96, -68, 2, -19, 81, -7, -50, -34, 24, 1, -75, 52}
}
, {{-16, 43, -83, 1, -108, -67, -82, -86, -12, 52, -54, -218, -8, -17, -112, -185, -26, -7, 17, -57, -101, -91, -140, 89, -3, 26, -23, -175, -13, -81, 76, 25}
, {19, 68, -45, 39, 50, 91, -71, -58, 85, 29, -114, -134, 5, 84, -15, 8, -104, -23, -148, 37, -20, 9, -79, 4, -7, 44, -51, -23, -58, -89, 8, 81}
, {26, -6, -68, -58, 54, 40, -67, -2, 15, -27, -90, 57, -14, 34, -17, 141, -69, -5, 0, -17, 18, 105, 13, -29, 4, 86, -63, 8, -29, -102, 82, 115}
}
, {{-212, 10, 117, 35, 22, -104, -180, -33, 138, -148, -190, -182, -54, 24, -26, -41, -256, -32, 25, 142, -74, 37, -29, -23, 109, 59, -186, -2, 56, -155, 43, 41}
, {-85, 67, -63, -128, -40, 63, -74, -7, 129, -58, -9, 46, -114, 136, -133, -38, -64, 75, -208, -17, 11, 9, -68, -57, 64, 132, -4, 121, -47, -74, -42, 99}
, {18, 35, 16, 113, 166, -118, 10, 3, -2, -16, 64, 148, -101, -2, -60, 21, 2, -80, 79, 24, 30, 40, 5, -62, 25, 54, 81, -12, 69, -66, 51, -10}
}
, {{126, -30, -74, -60, 80, 46, 41, 34, -18, 30, -69, 90, -14, 29, -47, 144, -15, -38, -56, -46, 7, 110, -18, -74, 6, -62, 47, 105, 48, -35, -38, 213}
, {33, 64, -56, -3, 21, -88, 29, 55, -20, 53, -41, -36, 69, 63, -81, 62, -70, 36, -13, -59, -25, 24, -14, -88, -107, -33, 39, 102, 33, -78, -2, -68}
, {-241, -66, 11, -84, -207, -44, -67, -136, 54, 3, 94, -205, -76, -21, -109, -50, -60, -161, -48, 100, -48, 15, -26, -3, 82, 112, -237, -16, -70, 145, 22, -181}
}
, {{-45, -23, -55, 2, 9, 72, -13, -28, 46, -53, 69, 0, -33, -52, -82, 51, 42, -39, 101, -3, 57, -34, 147, -51, -34, -76, 42, 57, 66, 141, -28, 37}
, {77, -45, -8, -5, -58, -46, -37, 32, 3, -2, -35, 64, -32, -53, -20, -69, -58, -38, 10, 35, -63, 43, -53, 24, -3, 14, -51, 9, 58, -14, 17, -32}
, {68, -38, 137, 50, -103, 170, -92, -82, -58, 103, -107, -45, 91, 15, -9, 159, 24, -108, -64, 45, 4, 40, -57, 49, 47, 46, -77, -66, -97, 142, -59, 82}
}
, {{-48, 55, 102, -72, -104, 30, -128, -77, -241, 240, -110, -46, 32, -103, -13, -72, 16, 1, -245, -55, -6, -194, -109, 8, -137, 136, -31, -46, -179, -70, -41, -8}
, {-123, -48, -34, -136, -88, -14, -101, -15, -132, 70, -118, 34, -8, 7, 37, 63, -56, 29, 10, -38, 140, -98, 95, -18, -91, 51, -39, 29, -14, 90, -72, 133}
, {39, -33, -13, 4, 43, -5, -31, 30, -2, -10, 29, 16, -22, 48, -6, 9, -142, 14, 8, -68, -43, -85, 77, -159, 133, 97, 174, 64, 142, -12, 122, -73}
}
, {{42, -79, 62, 38, 28, -50, -17, -16, -26, -102, -53, 55, 34, -41, -91, 8, -93, 13, -31, -76, 33, -76, 46, -81, 72, -38, 29, 16, 71, -12, 29, 20}
, {-66, -68, 54, -33, 36, -47, 8, 93, 60, 33, 19, 136, -43, -48, -110, 55, -138, -11, 16, -9, 35, -60, 21, 6, -59, 25, 95, -54, 98, -49, -33, 37}
, {-226, 18, 90, -29, -52, -19, -47, -27, 172, -40, 3, -78, -30, 56, -196, -67, 45, -180, 69, 94, -148, -110, 6, 15, -36, -62, -11, -23, 88, 0, -50, -71}
}
, {{-171, 87, 6, 9, -166, 13, -66, 87, -67, -149, -12, -160, -75, -88, 25, -106, -22, -9, -39, -70, 19, -16, -85, 73, -210, 59, 18, 19, 8, -88, 33, 91}
, {-5, 60, 2, -16, 105, -16, -40, 22, -6, 41, 16, 39, -3, -19, 42, -16, 153, -42, 2, -31, -41, -129, 47, -16, -89, -11, 56, 23, 31, 82, -51, -66}
, {-119, -66, -141, 11, 87, 67, -18, -123, 84, 21, 19, 18, -55, 16, 7, 14, 32, -121, 61, -13, -108, -24, -21, 127, 116, -32, -61, -74, -12, -43, 83, -46}
}
, {{19, -37, -143, 57, 57, -62, 21, -8, 87, -61, 108, 2, 12, 155, -109, 120, -208, -44, -6, 209, -6, 265, -27, 6, 147, 108, 22, -49, 14, -103, -13, 119}
, {-23, 91, 57, 43, 9, -113, -60, 109, -61, -40, -144, 62, 3, -75, -90, -103, -56, 26, 35, 119, -113, 68, -169, 63, 21, -76, -34, 50, -57, 3, -119, 35}
, {-99, 85, -73, -75, 52, -61, -65, 123, 72, -108, -21, 84, -22, 16, -356, 89, -232, -100, -221, 113, -37, 139, -16, 24, -26, -17, -20, 52, 71, 67, 65, 79}
}
, {{-98, 129, -33, -12, 3, 11, -40, 125, -22, -20, -97, 37, -43, -17, -31, 94, -72, -85, 22, -46, -44, -19, -99, 39, -83, -61, -36, 148, 84, -57, 5, 111}
, {30, -15, 82, 5, -54, 115, -97, -44, -19, -87, -31, -27, -46, -30, -57, 52, -16, 77, -139, -70, -17, -114, 120, -1, -95, 121, -76, 75, 51, -93, 82, -41}
, {-22, -144, -130, 47, -54, 32, 28, -244, 42, -55, 126, -178, -10, 111, 144, -61, 83, 79, 2, 10, 21, -51, -9, -143, 14, 63, 78, -108, -7, -78, 112, 30}
}
, {{-92, 63, 14, 101, -197, -38, -57, 7, -84, -34, 2, -64, 42, -18, -35, -3, -30, -54, -39, -10, -44, 129, -116, -174, -112, -8, 34, -85, -82, 40, -2, -16}
, {7, -50, 25, -20, -18, 24, -84, 1, -32, 73, -113, -6, 69, -43, 38, 91, 18, 99, -33, 4, -28, 52, 120, -70, 1, 6, -18, -5, 97, 95, 78, 52}
, {-284, -33, 70, 113, 28, -94, -91, -65, 3, 66, 92, -136, -4, 86, 105, 17, 30, -26, 67, 17, -4, -20, -16, -22, 66, 61, 41, 21, -29, 28, -116, -67}
}
, {{4, 43, 15, -65, 26, 131, -32, -17, 20, -81, -104, -18, -19, -84, -113, -29, -22, 32, -27, -96, 37, -57, 78, 82, 75, 46, -40, -31, 120, -71, 48, 53}
, {-85, -1, 69, -5, -84, 169, -44, -220, -14, -74, -78, -161, -37, -20, -21, 34, -3, 100, 63, -44, 74, -24, 102, 13, 107, -3, -52, -125, 35, 68, -5, 10}
, {-146, -52, -65, 35, -97, -25, 18, -175, -42, -80, -54, -27, -1, 20, 14, 2, -66, -12, -36, 14, -19, -16, 108, -22, -78, -12, -63, -140, 44, 4, 60, 77}
}
, {{-62, 22, -1, 19, -30, 74, -146, -7, 95, 1, -80, 40, 66, 42, -39, 111, -97, 0, 48, 106, -3, -54, 81, -48, 56, 16, -20, -12, 161, 99, 123, 68}
, {-49, 54, -107, -44, -61, 2, -103, -91, 26, -58, -120, 60, -77, -45, 29, -31, -174, 36, -3, -94, 11, -65, -102, 9, 18, 27, 29, -9, -24, -66, -25, -69}
, {28, -19, -71, 89, 55, 4, -127, 85, -20, -28, -37, -12, 41, 116, 27, 125, -168, 10, -18, 38, 0, 25, -35, -130, -74, 18, 37, -54, 15, 51, 5, 73}
}
, {{19, -98, 34, -83, 64, -29, -95, 25, -87, -85, -30, 116, 13, -88, -43, 18, 60, 75, 61, -115, 53, -163, 148, 109, -32, -112, 76, 89, 59, -85, 35, 86}
, {39, -46, 34, -83, -108, 49, 74, 29, -93, 26, -1, 68, 33, 4, 38, -24, 49, 78, -78, -155, 49, -94, 63, -113, -144, -34, -98, 34, 81, -2, 28, -29}
, {-85, -41, 41, 5, -96, -31, -63, -117, -99, -79, -102, -4, 26, 62, 8, 7, -78, -116, 33, -25, 5, -84, -16, 25, -29, 4, -156, -78, 8, 8, 19, -136}
}
, {{-142, 122, -54, -147, 34, 109, -41, -6, 127, -94, -82, -91, 45, 12, -65, 130, -16, -78, -232, 136, -25, -86, -98, -69, -38, 49, 85, -18, -39, -151, -60, 128}
, {-67, 0, 60, 50, 85, -12, -72, -4, -5, 40, -84, -19, -53, 34, 75, -102, 53, -166, 24, 0, 11, -99, -56, 28, 14, 49, -17, -107, -75, 1, 3, 35}
, {37, 21, 13, 39, 58, 75, -116, -163, 13, 68, -24, -55, -31, 117, 48, 56, 29, -91, -52, 93, 35, -60, -55, -35, 32, 80, -5, -152, -129, -26, -16, -64}
}
, {{-147, 1, -39, -25, -82, -51, -82, -91, -75, -167, 84, -42, -119, -66, 39, -213, -95, 105, 33, -86, 55, -108, 10, -21, -68, 7, -23, 24, 78, -39, 3, 8}
, {-135, -41, 20, 50, -6, -49, -72, -8, 6, -115, 54, 39, 27, -20, 24, -82, -51, 0, -3, 3, -36, -216, -20, 46, -54, -22, -7, -27, 215, 31, -97, -14}
, {5, -4, -14, -25, 87, 14, 70, 59, 16, -51, 37, 105, -35, -1, -2, -1, -81, -107, 104, -34, -69, -28, 108, -88, -12, -83, 91, 48, 89, -78, 145, 70}
}
, {{-44, 67, -144, -113, -118, -35, -41, -59, -76, -59, 45, -1, 33, 5, 3, -15, 4, -131, -45, -24, -19, -136, 22, 77, -149, 43, -82, -25, 17, 77, -7, -122}
, {-18, 58, -52, 6, -118, 74, 24, -47, -90, 32, -70, -40, 42, -27, -33, -65, 104, 63, -23, -79, 28, -115, 128, -7, -58, -143, -24, 48, -53, -173, -41, -70}
, {107, -3, 90, 86, -29, 136, 36, -83, -28, 30, -124, 61, -42, 20, 97, 24, -59, 90, -7, -35, 103, -4, -17, -152, -76, 22, -18, -39, 65, 83, -39, 2}
}
, {{-91, 47, 77, -59, -207, 20, -89, -29, -161, -104, -141, -178, -43, -60, -97, -153, 27, 36, -84, -47, 42, -67, 64, 36, -117, -31, -67, -31, -87, 85, -103, 184}
, {-9, 84, -24, -109, -78, 71, -53, -49, -41, 46, -102, 38, -71, 158, -29, -62, -94, -69, -57, 13, -23, -106, 35, 5, 69, -25, -59, -59, 97, -30, 71, -1}
, {-58, -105, 105, -156, -4, 9, -18, 75, -167, 30, -159, 71, -53, -105, -1, 96, 115, 1, -48, -206, 49, -45, 56, 59, -209, -46, 40, 27, -61, 99, -94, 97}
}
, {{-98, 8, -36, 20, -50, -24, 21, 37, -69, -28, 62, -23, 111, -129, -103, -137, 80, -80, 127, -49, -84, -161, 140, 30, -16, -7, -41, 40, -32, 87, 34, -29}
, {-1, -3, 11, 24, -2, 36, 109, -90, -6, 26, 120, 8, -19, 6, -69, 2, 43, -64, -7, 30, 59, -22, -67, 88, -94, -143, 63, -25, -79, 30, 5, -45}
, {-168, -1, -209, -132, 33, 26, 10, 56, 11, -115, 25, 8, 93, 171, -64, 39, -226, 31, -105, 4, -100, 158, 57, 81, 21, -45, 18, 59, 155, -29, 99, 62}
}
, {{-38, -76, 9, -111, -72, -55, -36, 58, -145, 34, -68, -61, -171, 116, -77, -112, -46, -3, 59, -34, -43, -84, 68, 74, -87, 20, -66, -60, -53, 176, -51, 0}
, {41, 71, -1, -72, -110, -40, -11, 189, 121, -12, -26, -27, -60, 10, 85, -3, -17, 12, 22, 156, 90, 56, -31, -98, 13, -11, -33, 50, 22, -22, 10, 18}
, {-68, 38, 48, -153, -103, -26, -38, 79, -31, 40, 31, 30, -2, -191, -76, -111, 13, -89, -69, -121, -42, 18, 24, 84, -21, -17, -37, 34, -3, 2, 34, 50}
}
, {{-96, -133, -65, 102, -55, -33, -63, -66, 39, 18, 48, -19, -47, -71, 60, -39, -43, -80, -48, 15, 54, -29, 21, -81, 5, 58, 2, -42, 42, -84, -37, 0}
, {-158, -1, -64, -77, 34, -35, 70, -11, 77, -163, -4, 82, 4, 31, 34, 16, -150, 10, 40, 35, -14, 64, 11, -15, 112, -1, 69, -45, 23, -86, 26, -12}
, {7, -34, -65, -23, 18, 59, -136, 28, 66, 2, -127, -4, 6, -3, -21, 34, -215, 62, 32, 61, -19, 79, 5, -86, 9, 51, -2, 75, 74, -174, 157, 16}
}
, {{-95, -53, 23, 89, -52, -66, -100, -139, -19, 99, -60, 0, 134, 66, 141, -136, -14, -126, 135, -3, 95, 22, -78, -10, -10, 126, 29, -206, -94, 144, -32, 9}
, {-4, -104, -26, 34, -74, 14, -19, 65, 19, 8, -35, -23, -65, 26, -22, -53, -112, 147, -19, 47, -38, -73, 56, -36, 34, 67, -50, -81, 134, -7, 19, 53}
, {98, -63, 52, -137, -48, 129, -45, 161, 69, 13, -49, 79, 29, -45, -87, 42, -66, -58, -25, 14, -6, -24, 102, 57, 29, -17, 57, 80, 93, -148, -4, 82}
}
, {{31, -42, -32, -244, 70, 85, 5, 109, -10, -78, 14, 105, -128, -164, -174, -17, -23, 72, 52, -53, -27, -51, 125, 6, 38, -264, 85, 95, 135, -44, 64, 90}
, {-47, -52, 44, -133, 57, 61, 14, 39, -77, 59, -27, 22, -49, -112, -110, 18, -26, -41, -23, -66, -101, -80, 88, 48, 6, -45, -28, 11, -59, -3, -26, 7}
, {-166, 70, 124, -101, -159, 25, -50, 12, -7, 97, -70, -168, 65, -22, -165, -54, 36, -227, -80, 69, -153, -88, -46, 53, 176, 36, -361, -222, -124, 128, -94, -90}
}
, {{-35, 27, 102, 31, -205, -5, -47, -16, -3, -128, 69, -65, 30, -60, 26, -12, 100, -41, 117, -42, 4, -62, 184, -6, 30, -1, 51, -100, -29, 180, -11, -18}
, {-56, 60, -150, 5, -60, -68, -19, 6, 29, -8, 35, -19, -104, -55, 58, 9, -76, -105, 6, -30, -4, -21, -18, -40, -4, 10, 22, 10, 72, -39, 28, 51}
, {-126, -55, -155, 32, 54, 12, 66, -87, 65, 6, 11, 52, -117, -66, 4, -76, -58, -63, 59, 69, -123, 45, 9, 33, 63, -45, -57, 52, -56, 25, 114, 6}
}
, {{55, -54, -3, -72, 73, 65, 30, 118, 46, -66, 69, 27, -4, 11, -25, 71, -18, 90, 49, 158, -67, -7, 20, -75, 149, 10, 69, 79, 155, -50, 21, 65}
, {-52, -35, 5, -75, -62, -11, 0, -26, 41, 17, -28, -145, 30, 3, 62, -54, 23, -183, -24, 29, -30, -61, -77, -53, 58, -19, -86, -8, 24, -81, 10, 54}
, {23, -4, 12, 0, 45, -49, 29, 84, -145, -11, 17, 95, 81, 78, 16, 35, 34, -6, 36, -30, -53, -12, -69, -109, -81, -40, 73, 47, -36, -73, -48, 36}
}
, {{-10, 75, -204, -155, -25, 92, 40, -77, -1, -11, -41, 120, -136, -8, -136, -56, 4, -27, -26, -145, -86, -104, -75, 45, -81, -29, -58, 107, -61, 61, 123, 80}
, {-28, 82, 52, 99, -63, -211, -87, 2, -65, 24, -119, 37, -140, 40, -92, 194, -142, -145, 27, 1, 24, 17, -26, -4, -75, 92, -36, -9, 41, -170, 88, -12}
, {-82, -18, -255, 37, -53, 133, -156, -25, 249, -60, -56, -25, -106, 275, -119, 60, -130, -132, -179, 114, 10, 114, -66, -153, 37, 149, -41, 44, -37, -175, 17, 95}
}
, {{-32, 3, 64, 54, 11, 2, -21, -139, 114, -38, -3, 66, -26, 85, 21, 24, -129, 21, -17, -37, 54, -13, -124, -176, 80, 72, -2, -67, -65, 31, -4, -7}
, {-18, 78, 38, 18, -13, -160, -15, 33, 45, -130, -59, 98, 26, -72, 91, -72, -84, -8, 91, 3, -12, 29, -49, -99, -34, 24, 34, -47, 74, -7, 42, -75}
, {-2, -102, -62, 34, -5, 135, -77, 13, 81, 3, -85, 95, -85, 33, -121, 75, -57, -99, -9, 17, -24, 23, -33, -113, 56, 92, 9, -40, 13, 34, -7, 104}
}
, {{-61, 3, -16, -5, 60, 20, -59, 46, -17, -1, -37, 183, -124, 20, 30, -61, -12, 45, 125, -128, 120, -44, 108, -61, -53, -81, 110, -103, 121, 57, -21, 40}
, {27, -61, 4, -56, -48, -72, 103, 107, -2, 20, -2, 30, -2, -136, -41, -39, -27, -90, -5, 11, -93, 42, -10, -2, -42, -2, 28, 71, -3, -83, -27, -76}
, {-90, 111, 40, -98, -126, 105, -10, -94, 6, 53, 60, -90, -8, 48, -25, -47, -5, -85, -102, -39, -49, 95, 13, 24, 89, 70, -153, 15, -89, -11, -89, -10}
}
, {{22, 74, -85, 43, -33, -90, -70, -153, 61, 26, -11, -126, 69, -115, 14, -131, -57, -152, -52, -88, -140, 39, -101, 56, -46, 114, 54, -55, 48, -86, 50, 88}
, {42, 80, -15, -49, 18, 54, 13, -31, 15, 27, -21, -21, -2, -103, 71, 59, 47, 11, -12, -32, -50, 83, -2, -42, -36, 5, -19, 54, 64, -128, 78, 20}
, {100, -106, 5, -82, 17, 124, -25, 152, 22, 63, 21, 100, 61, 54, 50, 119, -55, 64, 102, 86, 47, -108, 59, -87, -87, -137, 78, 63, 98, -18, -24, -56}
}
, {{-15, -65, -74, 16, 58, 41, -89, -21, 52, 111, -171, 111, -73, 16, 116, 53, -75, -86, -34, -43, 127, -81, -64, -123, 108, -22, 47, -27, -38, -97, -28, -39}
, {33, 66, -77, 103, 11, 53, -37, 79, 47, 52, -92, 60, -16, 78, 13, 32, -114, -101, 68, 1, 23, -94, -81, -11, -33, -74, -14, -6, -39, -98, 24, 5}
, {43, 91, -67, -83, 18, -29, -89, 132, 39, 186, -99, 21, 105, 24, -24, 4, -117, -22, -99, -10, 16, 11, 36, -107, -36, -33, 19, 44, 123, -86, -81, -16}
}
, {{-158, 4, -69, -116, -173, -72, 37, -67, 3, -96, 22, -89, -84, -81, -60, -196, -20, -18, -63, -53, -80, -127, 40, -31, -43, 2, -70, 73, 32, 47, 59, -52}
, {-84, 85, -13, -92, -37, -30, -57, -26, 20, -1, 71, -38, -26, -22, -57, -63, -59, -75, -6, -17, -126, -20, 98, -46, -68, 9, -61, -46, 115, -57, 56, 88}
, {-46, 33, 1, -140, -121, -11, 102, -4, -45, -8, 50, -98, -61, 3, -22, -102, -146, -26, -2, 64, -55, 90, 70, 78, 56, -54, -96, -30, 79, 103, 74, 36}
}
, {{105, 65, 69, -119, -228, 56, -89, -77, -2, 20, -71, -15, 30, -13, 73, -85, 57, 52, -82, -120, -76, -138, 16, 112, -164, -197, -108, -160, -44, -128, -23, -87}
, {59, 56, -19, -13, 1, 87, -70, 52, -120, 48, -122, -16, 151, -4, 65, 52, -31, -71, -74, 20, 20, 29, -37, -152, -122, -28, -128, -43, 32, -33, 16, 82}
, {-15, -226, -48, -154, 6, -22, -91, 26, -88, 109, -128, 19, 23, 22, -24, 57, -27, 54, -89, 65, 77, -31, -47, -64, -140, -2, -133, -45, -10, 15, -40, -106}
}
, {{41, -32, -81, -110, 6, 44, -58, 107, 12, -104, 12, 63, 7, 6, -88, 33, -66, -4, -72, 59, -46, -106, 186, -70, 70, -96, 31, 53, 57, -183, 72, 138}
, {-11, 96, -55, 57, -5, 61, 13, 97, 112, 11, -22, 30, -38, 7, -17, 13, -108, 8, -50, 47, -62, -12, -21, -14, 48, -3, 22, 24, -32, -129, 1, 170}
, {-399, 132, 17, 98, -21, -140, -53, -136, 75, -123, 72, -150, -259, 11, -84, 18, 158, -132, 25, 51, -147, -44, -101, 66, -5, 134, -95, 7, -53, 70, -104, -87}
}
, {{-76, 67, -23, 85, 136, -71, -104, -26, 129, 5, -4, -89, -42, 65, 9, 30, 27, -75, 0, 75, -62, 74, -95, -70, 99, 104, 4, 68, 7, -62, -19, 67}
, {-102, -14, -30, 73, -15, -56, -28, -103, -3, 84, 26, -95, 131, -14, -14, -62, 11, 14, -5, -58, -100, 14, -34, 89, 53, 20, -4, 13, -89, 110, 72, -27}
, {46, -9, -33, -17, -47, 182, 11, 147, 22, -13, -11, 14, 37, -145, -29, 28, 27, -4, 114, -98, 32, 18, -59, -30, -8, -46, 87, 29, 132, -20, 77, 57}
}
, {{-97, 137, 53, 53, -162, 48, -199, -84, -20, 163, -184, -57, -3, -29, 86, -181, -47, -161, -66, 11, -2, -71, -139, -3, 4, 56, 64, -140, -200, 105, -90, -10}
, {-23, -141, -72, -27, 72, 65, -19, -38, -17, 102, 33, -85, 4, -36, 0, 42, -13, -63, 79, -9, 85, -100, -23, -94, 35, -98, 59, -55, 30, 19, -6, 56}
, {-75, -92, -195, -10, 79, 19, 36, 158, 84, 5, 12, 81, -32, -37, -72, 27, 31, 169, -26, 1, 49, 24, 111, -64, 36, 16, 135, 99, 138, -208, -55, 222}
}
, {{-164, 138, 159, 13, -195, -87, -123, -200, 2, -102, -78, -79, 129, -35, 36, -36, -48, -116, -33, -17, -4, 28, -156, 49, -196, 47, -20, -130, -80, 92, -76, -77}
, {82, -17, 86, -169, 41, 12, -76, 6, -167, 110, -70, 55, -28, -54, -55, 78, 38, 12, 90, -76, 87, -73, 38, -5, -3, -125, 49, 53, -66, 115, -21, 53}
, {74, -49, -28, 3, 36, -90, -35, 178, -67, 35, -29, 132, 23, -94, -77, 6, 84, 65, 85, -157, -18, -70, 27, -77, -87, -129, 203, 3, 85, -50, -51, -43}
}
, {{-33, -19, -90, -36, 0, 30, -22, -3, 68, -26, 72, 2, 16, 82, -61, -2, -130, -4, -57, 93, 18, 26, -31, -19, 115, 87, -86, 64, 97, 30, 67, 14}
, {-54, -84, 7, 28, -72, 62, 12, -41, 81, -72, -70, -71, 77, 11, 17, -21, -23, 24, -49, 82, -74, 12, -59, -99, 108, 35, -37, -39, -10, -38, -5, 35}
, {-39, -39, 182, -45, -132, -71, -168, 177, -104, 41, -66, 13, -68, -11, -161, 30, -123, -110, 11, -39, -19, -20, -2, 87, -232, 99, -60, -140, 37, 210, -117, -73}
}
, {{-11, 110, 96, -19, 15, -83, -94, -37, 40, -51, 11, 6, -42, -12, -67, 151, 52, -63, 53, -89, 4, -76, 63, -25, -112, -96, -65, 49, -7, -16, -79, -41}
, {-84, 86, 141, -129, -172, 6, -168, -47, 79, -137, -10, 22, -105, -14, 20, -78, -71, 7, -19, -16, -109, 13, 1, 143, -13, -47, 15, 57, 28, 56, 16, -3}
, {21, 22, -36, 58, 33, 175, 34, 44, 41, -206, 79, 115, -42, 7, 50, -100, 141, 13, 86, -85, -70, -154, 133, 0, 53, 7, 171, -32, 145, -10, 257, 38}
}
, {{-26, -75, 138, -151, -17, 131, -63, 36, -67, 102, -33, -35, -35, -78, -123, 33, 168, 23, -94, -124, -85, -240, 140, -125, -150, -160, -29, 23, 30, 41, -55, -66}
, {53, -38, 62, -32, 53, -20, 34, -93, -127, 61, -21, 6, 45, 20, -8, -14, 118, -107, 13, -155, 2, -53, 109, 4, -32, -6, -9, -49, -51, 14, -138, -114}
, {29, -39, -76, 119, -45, -79, 42, -114, 63, 33, 85, -148, 84, 37, -14, 24, -38, -210, 32, -18, -20, 10, -18, -34, 43, 38, -215, -152, -180, 101, -76, -177}
}
, {{-107, 138, -67, 87, -117, -89, -171, -81, 91, -74, -27, -103, 97, 47, -50, -79, -153, -171, -13, 118, -34, 135, -217, -61, -3, 181, -112, -104, -59, -7, -54, 21}
, {7, -11, 15, -52, -58, -14, -119, 75, 33, -8, -44, -63, 72, 61, -126, 39, 11, -95, -70, -11, 4, 71, -36, -36, -92, -40, -23, 25, 35, 8, -78, 90}
, {136, -54, 27, -108, 47, 5, 59, 70, -30, -7, -15, 181, -37, -77, -155, 134, -1, -67, 46, -14, 6, -39, 122, -21, -105, -97, 119, 76, 119, -68, -49, 83}
}
, {{76, -117, 103, -118, 84, -71, -97, 67, -232, 107, -296, 140, 3, 15, 35, 150, 7, 1, 8, 4, 134, -35, 155, -127, -7, -145, -105, 77, -45, 93, -292, 155}
, {-112, -109, 36, -122, 40, -72, 38, 89, -277, 91, -150, 88, -57, -97, 38, 3, -3, 12, -34, -19, 8, -88, 25, 30, 100, 82, -100, -1, -54, 49, -45, -79}
, {-27, -25, 158, -88, -77, -21, 2, 91, -162, 101, -143, -28, -20, -163, -52, -22, 48, -158, -99, -90, -26, -117, -32, 52, -137, -14, -35, -10, -91, 114, -111, -43}
}
, {{-77, 95, 31, 2, -139, -45, 51, -40, 17, -165, -8, -19, 56, -63, 88, 23, 22, -32, 69, -33, 112, 136, -131, 25, -76, 83, -26, 102, 62, -85, 127, 73}
, {-90, -91, 9, 3, -123, -21, 52, 52, -26, -18, 66, 2, -14, 42, 61, 7, 11, 63, 44, -2, -60, 14, 134, -12, -76, 1, 80, 18, 33, 16, -9, 11}
, {-106, -126, -9, -99, -139, -56, 138, -7, -24, -108, 133, -23, -27, -137, 7, -28, 43, -18, 143, -4, -51, 22, 119, 94, -45, 57, -155, 98, 25, 65, 25, -113}
}
, {{51, -7, -24, 37, 94, -12, 35, 125, 3, 118, 54, 9, 94, -67, 32, 104, -26, -22, -36, -1, -69, 85, 91, -96, 13, -28, -56, 52, 115, -67, -53, -114}
, {35, 14, 96, -47, -47, 26, 46, 82, 31, -18, -41, 13, 28, -39, -17, -48, -81, -17, 45, 15, -40, 81, -65, 10, -28, 25, 9, 45, -29, 103, -17, 16}
, {-151, 77, 56, 113, 66, -36, 5, -101, 40, -85, -13, -90, 42, 61, 55, 21, -89, -115, 119, 49, 167, 29, -78, 111, 103, 110, -35, -128, -11, 33, -28, -78}
}
, {{72, -81, -4, 82, 117, 101, 59, -48, 61, 57, -7, 53, -45, 8, -27, -4, -28, 60, 80, -87, 28, -114, 56, 39, -25, 5, 48, -37, 141, -67, 1, 47}
, {-27, 63, 28, 63, 16, -92, 31, 101, -19, -47, 53, 32, -2, -38, -4, -16, 35, 78, -3, -11, -24, -7, -1, -44, 31, -43, 57, 36, 39, 84, 10, -52}
, {86, 6, 9, 105, -15, -14, 47, 43, -16, -44, 112, 21, 31, -52, -61, 31, 16, 51, 10, 8, -78, 23, -40, -48, -24, -26, 82, 153, -40, 65, -105, -73}
}
, {{40, -54, 41, 26, 56, -15, -62, -31, -86, 172, -106, 14, -86, -2, -17, -73, 57, -50, 2, -112, -5, -160, -26, 19, -197, -27, -9, -59, -24, 54, -193, -88}
, {28, -35, 26, 9, 45, -31, -90, -34, -69, 88, -123, 31, 53, 41, 13, 72, -29, -85, 84, -27, 85, -21, 73, -7, -94, 29, -2, -157, 26, 33, -146, -83}
, {-128, 32, -69, -139, 5, 7, -134, -101, -57, 29, -35, -93, -11, 49, 52, 54, -101, -110, 50, -65, -46, 15, 100, 13, 137, -21, -282, -85, -51, 5, 13, -132}
}
, {{-28, -177, 25, -39, -3, 69, -99, -74, 108, -85, 0, 147, -95, -40, -5, 59, -33, 81, 22, -227, 200, -55, -35, -26, 24, 28, 24, 77, 140, -222, 110, 65}
, {-122, 49, -53, -23, -44, -63, -68, 48, -8, -160, 6, 90, -16, -16, 50, -60, -110, -2, 30, 109, -3, -21, 58, -8, -26, -36, -34, -64, -80, 19, 0, -152}
, {-93, 151, -98, 5, -31, 55, -92, -14, 152, -56, 99, -37, -212, 110, -78, -9, -131, -56, -73, 150, 41, -23, -44, 54, 141, 61, 60, -15, -52, 64, -2, 91}
}
, {{-31, -163, 175, -49, -66, 59, 129, -53, -40, 113, 5, 0, 123, -61, 144, -44, 139, 24, 63, 49, 176, -5, 107, -116, -77, -8, 4, -42, -94, 131, -47, -3}
, {-23, -27, 115, -47, -62, -75, 61, 29, -43, -26, 95, 33, -9, 0, 141, 60, 44, 25, 41, 146, 35, 71, 63, 20, -58, -49, -16, 25, -62, 27, -61, -25}
, {-38, -75, 60, -50, 3, -61, 6, 112, -70, 17, -52, 86, -62, 100, 26, -64, -112, 50, -71, 62, -40, -55, 56, -41, 3, -12, 51, 68, 82, 9, 37, 26}
}
, {{33, -169, -86, 162, 201, -118, 17, 63, 120, 51, -20, 78, -10, -65, 0, 100, -10, 3, 120, 40, 38, 144, 74, -153, 127, -73, 36, 30, 44, -171, -75, 59}
, {-166, -32, -9, -42, -119, 74, -14, -101, 59, -67, -35, -101, -61, -59, -122, -192, -159, -11, 4, 26, -138, -8, -84, 50, 69, -15, -180, -68, 44, -107, -51, -74}
, {-67, 25, -9, -62, -181, 31, -92, -29, -44, -57, -57, -52, -46, 20, -42, -118, -36, -3, -50, 77, 9, -18, -115, -23, -36, 94, -65, -118, 43, 1, 10, -22}
}
, {{-41, -85, -28, 36, -96, -24, 18, -169, 21, 2, 19, -119, 115, -142, 50, 4, 16, -168, -25, -33, 97, 144, -107, -73, -46, 112, -94, -103, -115, -63, 17, 24}
, {-8, 0, 60, 30, -29, -35, -7, 11, 7, 31, 69, 123, 56, -38, -44, -37, -2, 13, -48, -15, 7, -38, -112, 49, -114, -33, -19, -20, -41, 25, -96, 49}
, {284, -62, -88, -29, -1, -25, 1, 27, 80, 48, -135, 90, 105, 29, -94, 34, -101, 23, -55, 22, -44, -1, 59, -91, -21, -41, 81, 97, 145, -72, 80, 82}
}
, {{55, 23, -4, 37, 10, -3, -32, 63, 52, -15, -5, 32, 60, 41, 22, -46, -63, -47, -30, 76, -106, 56, 98, 105, -46, -9, 27, -1, -8, 26, 68, 58}
, {-51, 31, -82, -93, -30, -22, -39, 21, 63, -137, 75, -46, -184, -102, 0, -2, 39, -12, -23, -39, -118, -16, 73, 66, 62, -58, -92, -14, -19, 79, -33, -12}
, {9, 104, 1, -64, -134, -132, 30, -194, 71, -114, 43, -45, -113, -161, -165, -175, 73, -1, -23, -79, -88, 19, 8, 66, -34, -18, -4, -3, -33, 17, -2, -120}
}
, {{97, 53, -72, -102, -7, 76, -124, 65, 140, -53, 9, 60, 147, 124, -57, -24, -10, -5, -144, 198, -55, -135, -25, -28, -62, 110, -81, -37, -94, -38, 3, 62}
, {93, 15, 133, -26, -87, 109, -165, -55, -103, -64, -129, 41, -51, -136, -49, 46, -107, 130, 50, -154, 83, -118, 7, 13, -19, -42, -47, -76, 105, 11, 72, -27}
, {-46, -29, 19, -70, -69, 51, -35, -100, 32, -202, 6, 89, -6, 137, -14, 59, -195, -58, -16, -90, 87, 126, 92, 16, 47, 65, -84, -110, 74, 26, 78, 37}
}
, {{9, 95, -56, -64, -60, -50, -108, -48, 77, 10, -45, 4, 109, 177, -62, -82, -29, -22, -96, 23, -94, 0, -184, -2, -14, 66, -62, 3, -113, -136, -2, 33}
, {62, 65, -20, 9, 45, -77, 26, -38, 75, -2, 55, 60, 22, -9, -21, 79, -59, -49, -68, 112, 27, 22, 2, -122, 51, -82, -25, 44, 12, -128, 111, 10}
, {125, -83, -31, 77, -18, 43, -65, 11, -28, 47, -58, -1, 129, 107, 80, 152, -141, 13, -14, -25, 27, 24, -55, -11, -1, 46, -96, -60, -43, 26, -16, -15}
}
, {{-222, 95, -123, 10, -15, 159, -79, -122, 6, -75, -46, -179, -84, 77, -21, -22, 34, -48, 57, -24, 61, -1, -60, -22, -44, 213, -8, -129, 18, -21, 63, 14}
, {-41, -51, -21, 49, 81, -43, 13, 83, -68, -2, -88, 69, 48, -8, -65, -110, 24, 39, 51, -84, 11, -66, 14, -10, -74, 16, 3, -107, 73, -10, 36, 38}
, {147, -19, 72, 3, -2, 54, -65, 77, 6, 31, -73, 13, 19, -109, -80, 49, 104, -30, -13, -17, -40, 3, 9, -11, -54, -49, 23, 58, -1, -73, -117, 62}
}
, {{-134, -18, -80, -11, 4, 87, -61, -7, -66, 57, -69, -10, 5, -3, 2, -56, 14, -21, 2, -12, 109, -81, 8, 27, -120, 39, 11, -14, -27, -42, 64, 55}
, {27, -39, -11, 73, -1, -25, -20, 58, -12, 32, 57, 24, -148, 86, 0, -18, 63, 43, 33, 14, 70, -86, 31, -3, 40, 90, -6, 56, 75, -127, -88, 45}
, {-215, 13, 94, 60, -127, -21, -124, 2, -86, 11, -17, 13, 23, -101, 43, 35, 254, 55, -11, -48, 81, -108, 92, -73, -174, -104, -2, 2, -27, 156, -155, -39}
}
, {{47, -70, -110, -30, 43, -73, 0, 46, -140, 31, -148, 114, -63, 117, -45, 3, 19, -143, -38, 57, -7, -2, 120, -37, -21, -71, 52, 27, -14, -74, -120, 66}
, {51, -61, -68, -2, 70, -6, -22, -70, -12, -32, -37, -94, -36, 153, 108, 126, -16, -72, 78, 86, 19, 32, 4, 48, 81, 94, 66, -12, -19, -157, 51, 65}
, {-65, -106, -118, 89, -11, -1, 54, -16, 85, 144, -136, 41, -41, 89, 16, -19, 14, 3, 40, -13, 41, -51, -30, -79, -5, 101, 132, 49, 83, -1, 97, 94}
}
, {{-81, 35, 35, -63, -35, -6, -80, -52, -56, 40, -31, -22, -30, -134, 98, 30, 104, -33, -73, 34, 52, -61, -126, 30, 1, 17, -107, -82, -134, 115, -3, 21}
, {-22, 5, -44, -143, -53, -18, -60, -114, -49, 36, 89, 59, 129, -55, -80, -3, 101, 50, 37, -56, 92, -84, 52, 65, -109, -12, -159, -22, 13, 16, 16, -214}
, {259, -132, 86, -57, -3, 119, 189, -9, -181, 105, -130, 29, 144, 34, 56, 58, 45, 69, -188, -116, 67, -4, -34, -218, -187, -134, -35, 89, -90, -20, -22, 66}
}
, {{-21, 14, 4, -71, 21, -11, -101, -111, -26, 90, 18, 22, -28, -74, -11, -2, 78, 57, 103, -69, 4, -154, 45, 5, 42, -107, 106, -96, 1, 89, 73, -68}
, {-57, -89, 88, -48, 56, 9, -52, 23, -71, 79, -47, -32, 3, -35, 0, -26, -69, -53, -57, -91, 63, -55, 61, 36, -74, 0, -78, 82, 65, 44, 67, -13}
, {-1, 64, 57, -39, -68, 47, -55, -11, -109, -10, 30, 16, -66, -44, 79, 5, 209, 78, 92, 21, 135, 47, 31, -5, -121, 15, 109, -34, -46, -144, -144, 5}
}
, {{44, 18, 17, 7, 108, 50, -45, -39, 59, -122, -77, 92, -97, 8, -81, 7, -19, 44, -22, 98, 39, -25, -53, -33, 23, 56, -68, -15, 77, -72, -53, 51}
, {33, -57, -194, 105, -8, 26, -143, 8, 58, -92, -54, 48, 16, 112, -142, 192, -245, -93, -111, 104, 2, 83, 69, -73, 51, 44, -44, -17, -35, -206, -71, 30}
, {-66, 91, -256, 53, -22, -19, -90, 93, 117, -23, -96, 56, -113, 161, -14, -139, -127, -108, -8, 91, 32, -13, -160, -164, 6, 71, 10, -76, 6, -196, -44, 66}
}
, {{-213, 81, 6, 38, -41, -117, -139, -132, 24, -54, 10, 29, 18, 39, -53, -98, 37, -113, 92, -42, -94, -93, 39, 26, -39, 73, -119, -140, -49, 207, 28, -74}
, {9, 105, -103, 70, -18, -13, -102, -25, -7, -126, -83, -9, 0, 24, -57, -138, -59, -68, -56, 37, -47, -92, -105, 52, 56, 49, 53, -62, 52, -26, 36, 93}
, {129, -54, -104, 17, 69, -13, -60, 56, 18, -59, -46, 33, 116, 128, 4, 58, -86, 49, -128, 30, -11, -13, 26, -139, -18, -62, 97, 129, 87, -224, -91, 137}
}
, {{96, -141, 68, -80, 133, 95, -15, 53, 38, -1, -51, 246, -10, 39, 43, 139, -89, 98, 75, 37, 47, -5, 168, -258, -9, -125, -47, -6, 154, -119, -46, -16}
, {-36, 4, -46, -10, 17, -19, -23, -5, 59, -18, 69, -75, 18, 55, 91, 60, 7, -57, -8, 20, 15, -34, 84, -48, 44, 14, 19, 21, -15, -41, -2, 22}
, {-217, -42, 6, -43, -138, -137, -65, -186, 0, 37, 37, -146, -98, 31, 90, -76, 27, -146, 65, -35, 19, -91, -148, 145, 88, 119, -152, -161, -142, 91, -70, 25}
}
, {{-150, 108, 50, -29, -97, -122, -62, -77, -18, 17, -16, -44, -109, 23, 69, -37, -2, -113, -17, 8, 35, -80, -76, -33, -93, 16, -138, -20, -106, 66, -68, -37}
, {-93, -25, 12, -56, -115, -129, -15, 88, -44, 46, 48, 31, -152, -34, 1, 12, 20, -87, -25, -6, 5, -78, 42, -50, -147, 5, -83, 106, -49, 66, -35, -8}
, {216, 32, -66, -89, -80, -35, 68, 57, -79, -38, 37, 23, 129, -56, -10, -16, 183, 64, -63, -136, -127, -73, 105, -44, -129, -92, 102, 132, 58, -47, 142, -116}
}
, {{-26, 54, 7, -45, -207, -164, 42, 107, -67, 15, 72, -188, -21, 26, -52, -46, 61, -32, -176, 72, -296, -102, 33, 71, -71, 93, -61, 224, -90, -34, -61, -1}
, {82, 87, 17, 61, -29, -118, 46, 18, -5, -21, 15, -83, 83, 9, -5, 60, 45, 118, -87, 7, -187, 40, 23, 116, -9, -112, 3, 169, -89, 34, -40, -7}
, {2, 68, 4, 107, 24, -98, 67, 3, -5, 39, 1, -92, 9, 25, 27, 9, -31, 2, -92, 95, -56, 50, -70, -24, 26, -38, -42, 11, -90, 26, -62, -1}
}
, {{-29, -173, -24, 5, 133, 125, 20, -26, -35, 45, -21, 36, 40, 40, 162, 43, -59, -63, 59, 27, 105, -38, 29, 53, 65, 121, 6, -107, 104, 3, 149, 49}
, {-68, 18, -14, 23, 29, -19, -31, 36, 32, -12, -30, 64, -108, 26, 44, -52, 23, -90, 52, 66, -22, -11, -11, 93, -26, -16, -23, -29, 19, -51, -20, 18}
, {-116, 104, 71, -43, -84, -184, 26, 73, -31, 73, 36, 65, 9, -189, -116, -172, 109, -31, 71, 65, -153, -52, -152, 13, -41, -69, -12, 110, -53, 4, -179, -40}
}
, {{20, -22, -86, -77, 102, 144, -33, 32, 39, -2, 6, 75, -10, 125, 26, 123, 5, 58, -14, -10, 29, -77, 76, -122, 7, -155, 59, 11, 61, -201, -52, 87}
, {-11, -59, -4, -56, 39, 60, -128, -65, -47, -22, 23, -60, 21, -14, 39, 15, 18, -66, -50, -128, -9, -13, -24, -41, -24, -43, -38, 34, -49, -46, 16, 80}
, {-15, -78, 195, 40, -1, 38, -103, -68, -82, 117, -122, -33, 115, 66, -30, 46, 59, 4, -71, -57, 52, 48, -118, -20, -37, 30, -53, -104, -81, 104, -55, 19}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_2_H_
#define _BATCH_NORMALIZATION_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       8

typedef int16_t batch_normalization_2_output_type[8][64];

#if 0
void batch_normalization_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_2_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_2.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       8
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_2_output_type output) {                // OUT

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
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_2_bias[64] = {-451, -146, -179, -317, -235, -299, -100, -288, -314, -111, 7, -231, -325, -225, -524, -277, -85, -384, -208, -335, -269, -324, -203, -92, -311, -190, -46, -62, -237, -108, -288, -80, 62, -212, -339, -166, -242, -326, -74, -189, -221, -348, -301, -498, -270, -274, -341, -70, -133, -390, -90, -194, -171, -55, -183, -110, -241, -146, -28, -270, -193, -340, -329, -128}
;
const int16_t batch_normalization_2_kernel[64] = {450, 872, 534, 553, 371, 515, 425, 767, 349, 525, 428, 487, 293, 512, 639, 484, 653, 725, 530, 1614, 346, 415, 853, 449, 289, 639, 341, 390, 297, 304, 1110, 1057, 590, 387, 396, 521, 663, 418, 634, 595, 501, 467, 324, 203, 1045, 456, 284, 537, 384, 794, 308, 294, 426, 464, 219, 455, 457, 498, 562, 435, 654, 346, 415, 456}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_2_H_
#define _MAX_POOLING1D_2_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   8
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_2_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_2(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_2_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_2.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   8
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_2(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _CONV1D_3_H_
#define _CONV1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       4
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef int16_t conv1d_3_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void conv1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const number_t bias[CONV_FILTERS],						                          // IN

  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_CONV1D_3_H_
/**
  ******************************************************************************
  * @file    conv.cc
  * @author  Sébastien Bilavarn, LEAT, CNRS, Université Côte d'Azur, France
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "conv1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       4
#define CONV_FILTERS        64
#define CONV_KERNEL_SIZE    3
#define CONV_STRIDE         1

#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0

#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void conv1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const NUMBER_T kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN

  const NUMBER_T bias[CONV_FILTERS],						                          // IN

  NUMBER_T output[CONV_OUTSAMPLES][CONV_FILTERS]) {                       // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short pos_x, z, k; 	// loop indexes for output volume
  unsigned short x;
  int input_x;
  LONG_NUMBER_T output_acc;

  for (pos_x = 0; pos_x < CONV_OUTSAMPLES; pos_x++) { 
    for (k = 0; k < CONV_FILTERS; k++) { 

      output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);


      for (x = 0; x < CONV_KERNEL_SIZE; x++) {
        input_x = pos_x * CONV_STRIDE - ZEROPADDING_LEFT + x;

        if (input_x >= 0 && input_x < INPUT_SAMPLES) { // ZeroPadding1D
          for (z = 0; z < INPUT_CHANNELS; z++) {
            output_acc += (LONG_NUMBER_T)input[input_x][z] * (LONG_NUMBER_T)kernel[k][x][z];
          }
        }
      }
      
#ifdef ACTIVATION_LINEAR
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
      // Activation function: ReLU
      if (output_acc < 0) {
        output[pos_x][k] = 0;
      } else {
        output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
        output[pos_x][k] = clamp_to(NUMBER_T, output_acc);
      }
#endif
    }
  }

#else


  static q15_t bufferA[2*INPUT_CHANNELS*CONV_KERNEL_SIZE];
#if INPUT_CHANNELS % 2 == 0 && CONV_FILTERS % 2 == 0
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_fast_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_fast_nonsquare(
#endif
#else
#ifdef WITH_CMSIS_NN
  arm_convolve_HWC_q15_basic_nonsquare(
#elif defined(WITH_NMSIS_NN)
  riscv_convolve_HWC_q15_basic_nonsquare(
#endif
#endif
                                      (q15_t*)input, //Im_in
                                      INPUT_SAMPLES, //dim_im_in_x
                                      1, //dim_im_in_y
                                      INPUT_CHANNELS, //ch_im_in
                                      (q15_t*)kernel, //wt
                                      CONV_FILTERS, //ch_im_out
                                      CONV_KERNEL_SIZE, //dim_kernel_x
                                      1, //dim_kernel_y
                                      ZEROPADDING_LEFT, //padding_x, left and right must be equal
                                      0, //padding_y
                                      CONV_STRIDE, //stride_x
                                      1, //stride_y
                                      (q15_t*)bias, //bias
                                      INPUT_SCALE_FACTOR, //bias_shift
                                      INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, //out_shift
                                      (q15_t*)output, //Im_out
                                      CONV_OUTSAMPLES, //dim_im_out_x
                                      1, //dim_im_out_y
                                      bufferA, //bufferA
                                      NULL //bufferB, unused
                                      );
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, CONV_FILTERS * CONV_OUTSAMPLES);
#endif
#endif


#endif
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    64
#define CONV_FILTERS      64
#define CONV_KERNEL_SIZE  3


const int16_t  conv1d_3_bias[CONV_FILTERS] = {81, -110, 58, -7, 23, 50, 37, 49, 142, -10, 38, 51, 37, 33, 28, 55, -35, 41, -34, 54, 126, 56, 39, 142, 157, 42, 10, 86, 138, 74, 64, 40, 85, 8, 91, 83, 70, 71, 40, 50, 60, -62, 70, 81, 52, 41, 26, 84, 19, 61, 88, -15, 27, 50, 18, 72, 79, 31, 18, 53, 56, 34, -44, 10}
;

const int16_t  conv1d_3_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS] = {{{34, 83, 176, -37, 19, -39, -96, -23, 196, 32, 21, 97, -3, 24, -47, -2, -37, -114, -18, -24, -3, 22, 53, -14, 45, 42, 95, 8, 2, 49, -54, -27, -88, 5, 12, -45, 1, -35, 120, 154, 18, -8, 54, 39, 100, -39, -26, 14, 37, -56, 128, 113, 93, 14, 9, -79, -124, 88, 15, 24, -93, 54, 2, 92}
, {-70, -7, 17, -77, 14, -17, 55, 12, 62, -28, 55, 54, 59, 49, 50, -6, 45, -53, -50, -56, -53, -38, -24, 31, -82, 106, 56, -46, -83, -33, -86, -55, -10, -37, -71, 47, 71, 22, 54, -6, -12, -18, -56, -16, -28, -106, 12, -41, -88, -91, 123, 38, -44, -40, -133, -103, 26, 22, -32, 39, -22, 14, -25, 1}
, {54, -109, 84, 67, 24, -125, -22, 32, 59, -5, -23, 2, 15, -43, -60, -73, -141, -83, 32, -75, -28, 18, 11, 17, -75, 166, 20, 11, -38, -2, -38, -108, -7, -78, -36, 69, 30, -9, -12, -37, -74, -35, -26, 12, -133, 34, 54, 18, -52, -75, 52, -3, 29, 16, -85, -70, 30, 143, -74, -25, -57, 19, 76, -71}
}
, {{-61, 0, -255, 148, 105, -119, 72, -136, -151, 23, -35, -135, -11, 116, -35, 102, -25, -96, -86, 108, 16, -80, 87, 65, 66, -82, -61, 0, -8, -53, -82, 171, 10, -110, -81, -230, -94, -95, 119, -66, 100, -15, -49, 24, 61, -2, 93, -277, -34, 62, -163, -74, -201, -37, 11, 40, 20, -90, 50, 197, 10, -74, -16, 5}
, {13, -85, -14, 38, 56, -42, -4, 107, 63, -109, -77, 95, 83, 4, 90, 24, -70, 104, -78, 50, 25, 20, 52, 10, -7, 98, 43, -117, -106, -84, -13, -20, 12, 38, -7, -17, 122, -54, -75, -78, 35, 29, 45, -41, -7, 65, 20, 77, 30, 89, 4, -34, 71, -45, -82, 100, 35, 109, -7, 68, 25, 56, -2, -80}
, {68, 109, 19, -34, 124, -85, 63, 35, -192, -9, 90, -10, 2, -46, 52, 0, 37, 135, -171, 8, -35, -26, -98, -7, -79, -30, 6, -5, 4, -26, 118, 183, 13, -45, -98, 16, 159, -259, 90, -79, 178, 40, -20, 61, -11, 65, -72, -53, -77, 41, 35, 110, -79, 69, -108, 168, -8, -22, -74, 26, 109, 27, 16, 20}
}
, {{34, 44, 2, -55, -32, 79, 22, -7, -48, 43, 91, 36, 38, 16, -114, 39, 42, -108, -23, -7, -38, 51, -120, 64, -20, 20, 58, -57, 51, -15, -25, 20, -35, -20, 47, 71, -32, 3, 65, 131, 115, 59, 35, -79, -57, 57, 111, -17, 45, 3, 62, 6, 13, 30, -56, 29, 26, -10, 48, -42, -61, 99, 15, -26}
, {-105, 17, 68, 0, 45, -24, 13, -1, 1, 64, -5, 23, 1, 40, 8, 8, 57, -66, -12, -22, 46, -46, 129, 33, 83, -6, 58, 37, 28, 21, -75, 50, 74, 35, -35, -58, 45, 48, 92, -26, 8, 52, 54, 18, 8, 77, 34, 14, -17, -71, 100, 47, -13, 103, -35, 59, -36, 31, 42, 108, -71, 30, 106, 78}
, {-84, 17, 11, -110, -6, 72, -73, 42, -19, -77, 69, -24, -28, -95, 21, 3, 75, 51, 60, -89, -8, -24, -32, -56, 26, -18, -2, -63, 42, 96, -61, 51, 54, 23, 13, 139, 51, -31, -61, 92, -43, -31, -17, 12, -179, -56, 43, -17, 64, -61, 20, 17, 75, 22, 11, -15, 50, 17, -2, -36, -62, 5, 56, -29}
}
, {{-90, -183, 125, 4, -69, -43, 15, 46, 82, 100, -62, 125, -19, -142, 146, -43, -45, -41, -121, -28, -48, -68, 151, -44, -4, 22, 52, -114, -111, 47, 44, -234, 118, 31, -17, -33, 37, 80, -119, 12, -218, -33, 6, -23, -1, 34, -168, 123, 13, 30, 4, -22, 15, -91, -82, -4, -59, 103, -58, 58, 13, -11, -68, -3}
, {39, -21, 63, -32, 33, 38, 30, 75, -59, -65, 30, 75, -48, -138, 83, 25, -16, 5, -45, 63, -131, 78, -63, 12, 26, 73, 5, 39, 4, 10, 60, 45, 58, 29, 42, 117, 99, 50, 3, -51, 61, 29, -38, 6, -49, -32, -14, -27, -9, 191, -18, -59, 1, -39, -60, 45, -6, -19, 75, 39, 176, 39, 37, 5}
, {43, -49, 55, 31, -49, 14, 28, 52, 56, 95, -15, 8, -100, -29, 2, 26, 8, -80, 54, -32, 77, -33, 20, 61, -3, -27, 21, -43, -26, -111, 68, -70, -65, 36, -2, 49, -70, 56, 50, 15, -36, 18, -76, -7, -57, -22, 63, 97, -95, -85, 40, 23, -28, -80, 49, -157, 28, 70, -47, -88, 36, 16, -56, -2}
}
, {{-86, -111, 24, 52, -12, -13, 41, 26, -18, -92, -62, 54, 7, 24, 42, 38, -12, 34, -15, 53, 45, 3, 35, -73, -28, 36, 55, -29, -32, -52, 19, -7, -15, -85, -111, -66, -5, 64, 8, -93, 113, 20, -25, 6, 91, 15, -65, -18, -79, 14, 71, -41, -71, -91, -44, -15, 43, 28, -26, 25, 50, -95, -80, 93}
, {77, -103, 21, -5, -60, 30, 8, -10, 41, 13, -49, 88, 15, 11, -32, -45, -60, 82, -55, 130, -40, 33, 39, -74, 49, 63, -1, -49, 39, -50, 108, -33, 83, -58, -36, -67, -16, -43, -70, 30, -93, -37, -33, -22, 37, -6, -69, -38, 8, 45, -38, 6, -58, -14, -7, -68, -70, -16, -34, -76, 34, 28, -72, -41}
, {50, -64, 40, 44, -34, -43, 29, 20, 17, -40, -66, 4, 81, 7, 7, 14, -82, -25, 3, 56, -27, -21, -23, -54, 39, 93, -7, 8, 38, 14, 49, -59, -7, -4, -7, -14, -75, -72, -15, 23, -87, -13, 56, 30, 14, 53, -14, 34, -56, 0, 29, -9, -2, -113, 3, -63, -32, 71, -57, 58, -99, -69, -21, 35}
}
, {{27, 54, -27, -51, -34, -24, 4, 34, -81, 80, 128, 21, 72, -113, -42, -55, -1, 41, 38, -27, -13, 27, 1, -80, -96, 44, -12, -4, 69, -26, 153, -55, -87, -27, 15, -39, -22, 17, -32, 32, 0, -24, -73, -11, -53, -9, -109, -46, 66, -23, 39, 116, 47, -37, -63, -180, 27, 22, -81, -114, -12, -82, -60, -16}
, {-19, -28, 34, 11, 3, -17, 2, -15, 70, 76, 13, 96, -64, 13, 61, -48, -59, 32, -66, -32, -60, -72, 70, -45, 22, 13, 38, 59, -53, -25, 122, -68, -3, -52, -39, -40, -29, -50, -24, -36, -133, -79, 20, -25, 8, 32, -27, 16, -16, 3, -4, -22, 29, -53, 1, -40, -63, 57, -91, -14, -7, -14, 26, -22}
, {109, -104, 32, 90, 48, 4, 0, 37, -30, -11, -15, 109, -66, -32, 24, -2, 1, 29, -150, 27, -85, -121, 26, 14, -7, -93, -60, 12, -114, 41, 92, -35, 107, 32, -61, -8, 34, 51, -33, -61, -10, 14, -5, 26, 107, 8, -1, 30, 50, 45, -10, -30, -12, 70, -118, 121, -5, -28, 81, 122, 74, 37, 41, -57}
}
, {{-90, 121, 54, -60, -50, 5, 46, 40, -29, 91, 76, 44, -7, -25, 91, 119, 38, 39, -1, -25, 22, -37, -57, 31, -176, 48, 11, -15, -61, 45, 102, 40, 47, -70, -51, 68, 12, -68, -1, -14, -14, -45, -107, -49, 38, 22, -50, -137, -7, -55, -15, -10, -17, 103, -114, -73, 40, 34, 33, -24, 2, -31, -52, 39}
, {-44, 30, 21, -41, -24, 58, -29, 38, -94, 22, -42, 4, 21, 14, 0, 105, 31, 37, 35, -35, -12, 58, -20, -11, -115, -31, 35, -39, -1, -7, 23, 121, 90, 58, -48, 42, 26, 81, 52, 43, -110, -29, -24, -113, 25, -10, -21, -35, 36, -63, 39, 27, -15, 67, -87, 55, 62, -90, 78, 25, 9, 16, 3, -55}
, {-102, 30, -80, -175, -134, -4, -18, 4, -25, -89, -131, -36, -25, 23, -63, 156, 11, -39, 133, -101, 56, 97, -104, -112, 47, -75, -2, -127, -10, 42, -111, 76, -13, -35, 65, 30, -132, 178, -133, 107, -86, 11, -196, -74, -93, -61, 71, -20, 43, -103, 16, -66, -7, -17, 4, -56, 55, -4, 29, -16, -4, -112, -94, -80}
}
, {{-50, -18, 39, 60, -67, -85, 82, -13, 171, 140, 113, 136, 93, -8, -16, 46, 97, 1, -42, 2, 107, -77, -114, -28, -49, 11, 30, -8, 82, 63, -42, 53, -35, -40, -86, -62, 114, 4, -32, 9, -2, 76, 59, 58, -76, 91, 36, 74, -38, -95, 20, 13, -27, 70, 25, 108, -35, 148, -39, -79, 29, 64, 33, -5}
, {-42, 33, 62, -37, 26, 109, -41, 18, 14, 122, 11, 89, 23, 31, 20, 92, 5, 6, -5, -27, -13, -50, 9, 99, -45, -31, 24, -113, -63, -22, -82, 2, 123, -9, -72, -12, 92, 33, 9, -21, 54, -4, -30, -10, -25, 80, 23, 31, -7, -78, 20, 1, -4, -21, -51, 51, 13, 11, 114, 131, -59, 70, 14, 38}
, {-122, 143, 44, -47, -111, 150, -61, 47, 72, 46, -23, 58, 46, -27, 7, 30, 4, -3, 68, -76, 94, 13, -17, 78, -3, -1, 60, -27, 25, 49, -123, 42, -43, 70, 72, 34, -12, 76, 1, 56, 25, -59, 16, -69, 8, 17, -22, 82, 37, -113, 12, 11, 51, -23, 133, -84, 11, 104, 11, -42, -94, -71, -36, -14}
}
, {{114, 43, 8, -30, -48, -61, -52, 25, -34, 75, 52, 50, -41, -21, 37, -73, -52, 1, 21, 79, -88, 26, 10, 105, -31, 48, 19, 13, 8, -6, 35, -28, -36, 15, -3, 49, 51, 35, -28, 96, 41, -140, -15, -88, 7, -153, -32, 39, -25, 39, 85, 17, -61, -96, -28, -7, 8, -28, 131, -1, 46, 156, 83, 18}
, {63, -9, 45, -75, 32, 15, 12, 34, 19, -30, 63, -73, 13, -54, -65, -56, -21, 9, 7, 2, -19, 45, 1, 42, 55, -24, -12, 24, -90, 68, 15, -52, -51, -44, 18, 36, -92, -36, -2, -3, 17, -22, -37, -15, -89, -34, -26, -41, 14, 122, -69, 5, 55, -46, -59, 40, -33, 25, 42, -201, 53, 7, 34, 30}
, {24, -77, 96, 86, 107, 25, 8, 76, 30, 30, 65, 6, -18, 117, 7, -138, 5, 13, -8, 87, -60, -153, 37, -54, -12, 90, -8, 32, -14, -16, 62, -125, 36, 36, -37, -47, -37, -46, 139, -40, 43, -109, 136, -2, 93, 3, -80, -20, -10, 171, -2, 129, -67, 47, -45, 137, -54, 50, 43, -33, 88, 192, 13, 92}
}
, {{40, 3, -17, -53, 33, 72, -47, 49, -128, -36, -11, -55, -61, 91, -17, -124, 16, -16, 46, 9, 24, -20, -23, 56, -145, 29, -2, 49, -99, -18, 72, 95, -82, -38, -44, -27, 69, -28, 149, 30, 103, 10, -85, 57, 72, -87, -30, -26, -54, 14, -30, 15, -85, -2, -42, 10, -22, -14, -2, -86, -44, -91, 6, -78}
, {134, -35, 6, 20, 26, 52, -61, 96, 105, -41, -11, -32, -40, -80, 126, -15, 35, -26, -51, -10, 65, -19, 26, -37, -11, -22, 9, -8, -58, -17, 42, 10, -70, 36, 4, -44, -36, 46, -24, 83, 22, 11, -29, -34, 81, 47, -41, 71, 10, 37, -17, -2, 20, -44, -74, 29, 30, 96, -17, -74, 50, -20, -71, -70}
, {1, -4, -94, -147, -47, 30, -38, 28, -85, -49, -39, -79, -1, 7, 135, -12, 34, 59, -66, 29, -34, 13, -66, -34, -24, 123, -85, -99, 48, -77, 58, 10, -7, -60, -130, 109, 68, 3, -104, 105, 54, -57, -27, 16, -130, -34, 9, -48, 53, -16, -63, -39, -104, 68, -91, 137, -17, -43, -55, -119, 20, -65, -93, -52}
}
, {{-43, -119, 37, 45, -38, -72, 10, -31, 45, 41, -45, 14, 25, 233, -51, 6, -5, 18, 71, -16, -5, -39, 138, -96, 35, -14, 29, -59, -81, -63, 29, -64, 192, -102, -103, -29, -10, -36, 53, -119, 25, 25, 7, 35, -33, -5, -17, 5, 23, 46, 15, -13, -45, 33, 28, -28, -49, 3, -46, 117, -110, -40, -75, 97}
, {35, -87, 51, -71, -27, -43, 23, 54, 87, -57, 56, -16, -31, -76, 19, -42, -14, -10, -98, 61, -23, -71, -19, -54, -18, 17, 0, 25, 1, -23, -130, -71, 11, 39, 23, 14, 52, -54, -81, 12, 53, -6, 113, -13, -5, 6, 8, -35, -40, -11, -103, 15, 64, -21, -18, 26, -52, 21, -37, 21, 26, 28, 69, -51}
, {54, 7, 29, 70, -102, -53, 19, 128, 116, -76, 89, -56, -4, -42, -16, 18, -117, -62, 5, 93, -31, -24, 77, 52, -113, 103, -29, 77, -149, -107, 46, -24, 88, 6, 42, 33, -1, -31, 2, -20, 95, -32, 3, -84, -18, 4, -106, 25, -145, 83, 10, -56, -9, 72, -93, -126, -16, 124, -67, -64, -65, 1, -16, -15}
}
, {{-14, -82, -61, -14, 11, -117, -51, -122, 23, -10, -49, 29, -29, 41, 70, 7, -51, 142, -39, -81, -52, -76, 100, -4, -1, -65, 4, -80, 12, -10, 57, -101, 127, 39, -85, -75, 22, 2, 8, -29, -4, 2, -98, -55, 26, -70, -103, 52, -55, 5, -2, -38, -126, -33, 36, -40, -5, 10, -82, -51, -35, -127, -168, -14}
, {25, -45, -11, -23, 59, 29, 12, -78, 25, -31, -19, 53, 15, -20, 20, 91, -3, 38, 23, 52, 70, 12, 15, 23, -49, 121, -43, -80, -12, -21, 72, 14, -35, 6, -32, 6, 64, 0, 99, -19, 65, -22, 1, -9, 30, -3, 65, -70, -9, 19, -36, -17, -65, 86, -14, 10, -70, -13, -9, -4, -45, 39, -131, 10}
, {21, -11, -76, -64, 12, 85, -42, -29, 52, -3, -16, 38, -23, -7, -30, 40, -118, 47, -30, 4, 26, 7, -59, 6, 42, 66, -69, -106, 9, 3, 37, -20, 81, 14, 60, 13, 40, -18, -71, 7, 150, -50, -91, -107, 36, 35, -32, -48, 27, 13, 10, -18, -30, 43, -22, 63, 9, 17, 4, -53, 29, 25, -139, 34}
}
, {{-13, -78, 53, 142, 32, -60, 34, -46, 63, 55, -65, 43, -58, 123, -3, -58, -6, -46, -11, -58, -97, -156, 93, 11, 177, -73, -71, 66, -166, 1, -17, -85, 99, -78, -157, -151, 19, -33, 117, -86, -116, 33, 74, 3, 98, 92, -50, 79, 23, -65, 4, -66, -68, 14, 49, -103, -102, -30, 5, 181, -124, -52, 87, 46}
, {-12, 39, -80, -20, 36, -35, -2, 36, -31, -7, 12, 110, -55, -15, 6, -13, 0, -2, 70, 34, -80, -54, -66, 37, -95, 101, -103, 1, -15, -44, -56, 32, -69, -22, -54, -61, 8, -3, -57, -18, -11, -38, 43, -21, 22, -61, 16, -4, -61, 60, 43, 1, -16, -56, -26, -29, -32, 62, -10, 78, -104, 103, -1, -19}
, {94, -54, -19, 42, -59, -107, -70, 149, 21, 134, 88, 208, -60, -5, 0, -61, -33, -26, 29, 7, 7, -6, 19, 56, -181, 24, -50, 61, -169, -130, 70, -14, 7, -63, -95, -98, 5, -151, -55, -51, -105, 32, -66, -41, 63, 25, -25, 20, -27, -58, 121, 72, -40, -72, -10, -4, -53, 52, -76, 48, -42, 113, -106, -9}
}
, {{-35, -10, 147, 34, 56, -28, -32, 91, 113, 30, -75, -1, 88, -29, 34, 10, -124, -79, -127, -51, 8, 26, 71, -152, 33, 186, 72, 28, -138, 50, -67, -198, 86, 94, -74, 116, -36, -9, 22, 6, -319, -43, -16, -8, 175, 26, -195, 137, -138, -75, 24, -45, 97, -68, 27, -99, -187, 216, 43, 28, -69, -76, 6, 75}
, {78, 58, 39, 1, -63, 72, 18, 79, -86, 34, 4, -37, -92, -36, 42, -53, 22, -45, -15, 47, 23, 50, -41, 43, -62, -28, -19, -29, 5, 9, -72, -42, 30, -50, 36, 16, 83, 48, -175, -1, -100, -48, 46, -76, 12, -59, 10, 52, 90, -75, 41, -21, -4, -31, -13, -116, -46, 20, 56, 64, -15, -29, -46, 10}
, {2, -67, -41, -105, -30, -37, -32, 144, 64, -134, 16, 20, 2, 95, -17, -27, -76, 11, 61, -101, -93, 59, -43, 60, -8, -30, -30, -33, 45, -27, 0, -36, -137, -92, 20, 117, -109, 183, 77, 91, -179, 63, -119, 112, 25, -84, 197, -27, 79, -93, 8, -29, -73, -38, -38, -116, 92, -33, 22, -123, -47, -10, -51, 18}
}
, {{74, -73, -37, -59, -12, -55, 14, -9, 20, -30, 8, -23, -143, -84, 29, 4, -14, 49, -72, 37, -27, -23, -76, -42, -62, -17, -104, -121, 41, -115, 118, -24, -68, -11, 109, 1, -12, 33, -85, 7, 30, 50, -45, -103, -158, -33, 44, 17, 65, -47, -110, 37, 4, -35, -19, -20, -46, -8, -35, -129, -58, 86, -83, -58}
, {-10, 8, 17, 32, 89, 5, 5, 18, -24, -52, 79, -19, -111, 114, -107, -98, -49, 105, -54, -15, 51, -22, -6, 38, 27, -77, -86, -10, -31, -17, 6, -18, 84, -65, -50, -45, 46, -21, -15, 9, 88, 1, 20, -36, -23, -37, -11, -25, -18, 31, -100, -11, -60, 26, 82, -8, -28, -26, -16, -83, -55, -33, -30, 23}
, {58, 23, -9, 141, 21, -69, 35, 26, 0, 26, 41, -9, -104, 52, 28, -128, -42, -42, 91, 40, -55, -87, 104, 94, 2, 23, -12, 145, -58, -96, -22, 4, -23, 86, -15, -54, 141, -177, 112, -21, 79, 92, 88, -24, -22, -139, -34, -6, -91, 7, -8, -21, -36, -126, -35, -66, -143, -59, 23, 21, -24, 3, 106, -26}
}
, {{81, 66, 103, -13, -22, 112, 10, 28, 22, 27, 12, 27, 52, -30, 4, -31, 26, 22, -85, -71, -12, 37, -30, 51, 18, 60, 84, -17, 30, 58, 3, 70, -42, 31, 5, 9, -25, 65, 60, 44, 58, 3, 14, -21, 13, -18, 47, -97, 85, -34, 31, 14, 25, 64, 122, 37, 28, 21, 3, -25, -63, -45, -50, 29}
, {-64, 57, 10, 39, -3, 34, 83, 72, 15, 93, 33, -41, -36, 15, 15, 46, 116, -27, -47, -9, -18, 22, 93, 104, -47, 34, 28, 24, 72, -33, -92, 16, 0, -7, -4, -49, -30, 59, -12, 21, 60, 22, -11, -4, -65, 31, 73, -44, 26, -78, 98, 7, 79, 61, 6, 5, 21, 38, 44, 76, -70, -82, 96, -35}
, {22, -7, -85, -42, 35, -38, 12, -7, -4, 57, 3, 82, 25, 45, -3, -25, 103, 16, 39, 30, -82, 56, -33, 36, 0, -37, 46, 41, 64, -64, -79, 9, 74, 57, 33, 25, 45, 47, 89, -12, -8, 77, 24, -35, -18, -18, 72, -46, 44, 47, 75, -25, -37, 10, 25, 95, 109, -45, 54, 85, 10, 11, 87, 88}
}
, {{-32, -76, -102, 0, -25, -68, 13, -84, -142, 25, -87, 106, 12, 166, 9, 158, 27, 13, 42, 59, 17, -136, 67, 1, -12, 74, -72, -31, -63, -82, -5, 26, 21, -94, -88, -40, 24, 11, -39, -199, -13, -38, -131, -47, -29, 8, 22, -137, -71, 12, -183, -113, -128, -91, -61, -32, 16, -6, -76, 25, 13, -26, -58, 17}
, {39, -82, -25, 26, -77, 91, -19, -73, 15, -61, -40, 5, 37, -20, 44, -12, -37, 53, -51, 42, 84, -15, 32, -93, -51, 113, 17, -1, -11, -3, 2, -97, 31, -33, 30, -22, 90, -12, 11, 22, -18, -34, 11, 33, 17, 1, -69, 21, 22, 6, 46, -18, -32, -94, -2, 22, -80, 61, -56, 96, 129, -37, -62, 32}
, {-31, -2, -13, 77, -59, -106, -13, 48, 51, -55, 85, 20, 2, -61, -37, 32, 4, -7, -87, 111, -73, -46, -1, -9, -89, 102, 48, -39, 11, -58, 84, 115, 65, -57, 8, -19, 63, -148, -141, -26, -70, 114, 49, -56, -16, 40, -47, -16, -37, -27, 13, -11, 74, -65, -37, 55, -138, 90, -78, 74, 140, 21, -15, -35}
}
, {{-59, 89, -154, -31, 38, 45, -77, -76, -137, -48, 86, 112, -164, -15, -87, -26, -44, -22, 9, 89, -107, -66, 134, 70, -42, -36, -20, -17, -43, -94, 33, 45, 69, -74, -77, -3, -24, -9, 134, -107, 79, -55, -62, -100, 26, -79, 47, -144, -30, -32, -52, -135, -141, -72, -1, -3, -6, -166, 103, 99, 15, 14, -2, 90}
, {-13, 35, -86, 37, 63, -17, -95, 63, -122, 22, -45, 68, -78, -57, 44, -30, -103, -6, 11, 14, -30, 37, 35, -78, 19, -57, 23, 1, -61, -34, 38, 39, 12, 36, -48, -9, 28, -64, 33, -72, -103, -18, -31, -79, -25, -25, 5, -24, -98, 58, 66, -5, -63, -29, 32, -28, -7, 8, 47, 45, 29, 69, -2, 34}
, {-18, 43, 51, -26, -77, -56, -90, 64, 102, 120, -18, 80, 19, -45, 3, -3, -110, -16, 96, -3, 67, 60, -39, 63, -17, -21, 101, 8, -84, -34, 7, -154, -125, 105, 89, -35, -144, -97, -18, 37, -234, -48, -33, -108, 13, -36, -2, -3, 39, -36, 70, 57, 46, -172, 83, -181, -46, 90, -74, -3, -151, 93, -64, -91}
}
, {{51, 0, -11, -84, 29, -17, -58, -3, -93, 33, -21, 12, -73, -77, -3, -71, 73, 26, 40, 62, -27, -61, -27, -57, -30, 11, 6, -65, -23, -74, 11, 126, -25, -9, 75, 2, 54, 0, -77, -33, 6, -72, -21, -68, -5, -100, -69, -129, 39, 13, -28, -24, -34, 7, 12, 52, -39, -16, -47, -81, 70, -31, -66, 6}
, {37, 3, 31, 67, -21, -6, -91, 18, 2, -7, 44, -1, -36, -82, 38, 50, -24, -67, -20, 21, -8, -29, -29, -18, 12, 14, -64, -68, 10, -8, -8, -28, -29, -54, 20, 13, -97, 53, 36, 48, -42, 18, 0, 9, -78, -55, 0, -8, -7, -16, 66, -42, 21, 17, -15, -9, -72, -2, 31, -83, 7, -11, -34, -39}
, {40, 8, -80, -36, 16, -29, -64, 82, 48, 47, 3, 17, 40, 59, -65, 70, -2, -1, -10, 69, -9, 26, -34, 19, -4, 15, 21, -54, 13, -22, -27, 29, 139, 43, 12, -48, -18, 103, -18, -48, 107, 91, -32, 51, -56, -8, -2, -65, 86, 185, -31, 82, -72, 63, 1, 62, 29, 29, 91, -2, 207, -26, -13, -38}
}
, {{11, 95, -13, -7, -19, 101, 87, 8, -9, 43, 38, -71, 20, 76, -26, 60, 75, -44, 31, -6, 122, 12, 51, 4, 29, 11, -5, 55, 103, 36, -25, 65, -18, 40, 64, -56, -73, 20, 134, 71, 31, 36, 58, 52, -10, 23, -36, -43, 26, 23, 16, 37, 45, -7, 53, -42, 63, 14, 48, 23, -49, 22, 97, 45}
, {-29, 116, 41, 26, -13, 16, 12, 43, -9, 56, 53, 26, 4, 46, 18, -23, -11, -114, 61, -6, -54, -43, 62, 140, 25, -34, 37, 48, -16, -32, 4, 39, -20, 37, 5, -13, -55, 8, 134, 37, 98, 72, 11, -2, 9, 3, 35, -38, 36, 11, -7, -24, -45, 35, 73, 16, 57, 34, 70, 1, -77, 12, 48, -56}
, {17, 57, 16, 94, -3, 27, 44, 86, -10, 94, 40, 52, -82, 24, -63, -41, 22, -94, 2, 30, 2, -75, 52, 88, 0, -53, 39, 69, -103, -98, 88, 62, 7, -41, -86, -88, -26, -39, 98, -52, 41, 133, -1, 64, 66, -65, 9, -4, -67, -25, 56, -43, -19, 47, -17, 53, -5, 9, 51, 137, -39, 50, 107, -1}
}
, {{111, -20, 37, -10, -38, -86, -33, 5, 41, -50, 95, 148, -88, -133, 49, 67, 47, 5, -18, 14, -62, 45, -92, -6, -73, 9, 32, -18, 51, -116, 127, -60, -36, 74, -18, 17, 35, -91, -147, 33, -15, 10, -49, -124, -30, -19, -7, 70, 81, -2, 46, 45, 65, -5, -132, -4, 12, 78, 22, -133, 95, 326, -33, -170}
, {-74, -63, -5, -70, 61, 34, -81, -30, -55, 4, -41, 100, -31, 27, 8, -30, -80, 29, -7, 98, -3, -7, -13, -142, 48, 49, -23, -81, -29, 64, 20, -20, 24, -51, -60, 37, 67, 32, -19, -29, 14, -125, -46, 18, -5, -9, -65, -75, 18, -34, 12, 13, -115, 40, -73, 43, 19, 1, -20, -72, 23, -47, -101, 24}
, {-49, -61, 7, -14, -13, 105, -47, 1, -47, 44, 59, 155, -19, -77, -116, -29, -21, 83, -3, 7, 36, 72, -38, 74, 45, 7, 32, -1, -7, 107, 22, 16, 67, 24, 47, 83, -17, 71, 10, 4, 73, -45, 21, -22, 20, -64, 23, 1, -47, -58, 67, -133, 29, 108, 68, -22, 1, 22, 48, 130, -48, -27, 30, 20}
}
, {{4, -12, 12, 30, 35, 41, 12, -79, -52, 114, -38, 82, 85, 69, -52, -49, 30, -27, -17, 14, -2, 16, 56, 109, -114, 83, -35, 36, -20, -74, 28, 31, 3, -61, -69, -66, -29, 41, 42, 5, 103, -48, -89, 50, -118, -26, -14, -57, -61, -67, -72, 62, -27, 16, 3, -116, -45, 50, -7, 11, 31, -4, -80, -33}
, {-73, -6, 69, -52, -35, 76, 6, 142, -51, 42, -50, -1, -7, 47, 13, -57, 14, -42, 31, -16, 0, -19, 28, -3, -7, 31, 37, -5, -57, 44, -15, 96, 91, 2, 62, 163, 28, -31, 17, 122, -98, 44, -7, -87, -98, 1, 12, -19, 94, 55, 80, -8, -15, 26, -25, 54, -68, -39, 36, -97, -19, 74, 7, -40}
, {-26, -113, -175, 22, -43, 16, 2, -81, -164, -106, -72, -61, -37, 70, -76, -30, 40, -55, -24, -51, -62, 53, -85, -16, -17, -35, -13, 23, 27, 46, 2, 53, 17, -40, -16, 26, -29, 45, -67, -47, 73, 114, -82, -10, 19, -103, 7, 18, 16, 69, -37, -124, -25, -20, -81, 123, 101, -149, 38, -28, 59, -50, -63, -2}
}
, {{-19, 74, -110, -81, 67, 72, 34, -157, -126, -49, 47, -58, 47, -1, -90, -118, 55, -27, 120, 60, 0, 8, 12, 134, -104, 37, 35, 61, 36, -10, -15, 78, -24, 19, 75, -55, -36, 33, 64, 29, 136, -74, -14, -55, -31, -168, 85, -128, 16, -7, -28, 39, 59, -11, 78, 5, 54, -13, 105, -5, -30, -3, -61, -90}
, {-39, 0, -21, -74, -34, -9, 50, 9, -14, 0, 76, -4, -29, -53, 14, 6, -31, 30, -11, 60, 7, 1, -14, -1, 19, -7, 20, -86, -89, 4, 22, -176, -62, -6, -3, -15, -72, -22, -68, 55, -38, -82, -79, -7, 52, -41, -53, -56, -12, -77, -76, -70, -22, -20, 28, -88, -106, 24, -78, -143, -31, -100, -59, -62}
, {38, -102, -4, 200, -103, -102, 32, 3, 39, 58, -102, 34, -4, -114, 3, 1, -73, 136, -194, 95, -25, -59, 80, 18, -174, -28, 80, 27, -168, -42, -22, -69, 61, -55, -101, -49, 81, -208, -140, -145, -85, -20, 62, -89, 8, 87, -120, -17, -207, 79, -8, -39, -20, 6, -94, -23, -166, 35, -127, 13, -49, -61, -19, -54}
}
, {{-54, -72, -7, 20, 10, 9, 69, -82, -47, 17, -24, 5, -9, -36, 72, -36, 47, -45, 106, 23, 6, 93, -70, 82, 19, 5, 44, 70, -24, -34, -101, 75, 31, -34, -111, -121, 132, 86, 39, -12, -11, -48, 77, 35, 14, 85, -16, -60, 53, -105, 22, -15, 23, 26, 61, -25, 61, -56, 104, 114, 36, 42, 139, 81}
, {-89, 71, 19, -55, -64, 38, 34, 46, -1, 6, -17, 20, -16, -13, 45, 28, 13, 4, -7, -50, -86, -20, -18, 39, 29, -7, 22, 94, -18, 26, -43, 16, -120, -36, -21, -49, -64, -1, -102, 40, -53, 49, -60, -6, -79, -69, 62, 69, -45, -29, 27, 55, -33, -3, -14, -74, 7, 38, 25, -110, -40, 2, 50, 46}
, {55, -26, 22, 102, 127, -30, 151, 5, 4, 21, -3, 25, -95, -79, -8, -107, -35, 55, -114, 121, -61, 71, 102, 75, -62, 59, 45, 182, -84, -95, 17, -30, 53, -30, -109, -54, 82, -118, -28, -42, 132, -86, 76, -1, 28, 24, -52, 107, -125, 7, 31, 45, -3, 56, 35, -49, 19, 20, -55, 77, -26, -91, 141, -14}
}
, {{-18, -137, -16, -9, -27, -51, 61, -136, 1, -147, 17, -31, -3, 3, -6, -38, -75, -32, -24, 38, 47, 10, 31, 103, -74, -100, 114, 9, 11, -37, 10, -19, -49, -83, -128, -33, 113, -38, -15, -86, 176, -33, -61, -99, 71, -29, 24, -21, -114, 32, 9, -124, -79, -5, 74, -33, 147, -96, 39, 157, -40, 41, -100, -9}
, {14, -44, 20, -8, 22, -15, 93, 42, 38, 16, -9, 53, -69, -41, -46, 29, -31, 50, -22, -10, 3, 37, 96, 18, -5, -110, -50, 34, -119, 44, 36, 52, 30, -76, 41, -56, -45, -84, 136, -28, 20, -15, -29, -92, 35, -17, -66, 2, -8, -11, 75, -17, 82, 14, -10, -59, -45, -73, -55, -12, -75, 123, 67, 45}
, {-5, 66, -98, -50, 36, 198, -34, -15, -37, -54, -33, 1, -34, -21, -105, 25, 84, 87, 31, 20, 56, -21, -5, 26, 107, -41, 5, -72, -20, 131, -8, -2, 6, -36, 65, -20, 14, 43, 51, 40, 103, -159, 20, -103, -46, -40, -13, -3, 14, 54, -77, 19, 30, 60, 56, 51, 9, -13, 134, 42, -66, -91, 16, 27}
}
, {{10, 37, 13, 9, 15, 34, 15, 83, -28, 91, 65, 105, 66, 5, -115, 128, -1, -15, 77, -35, 33, 112, 21, 54, 27, -67, 85, 45, 12, 37, -15, 47, -17, 95, 70, -59, -36, -9, 37, -113, 60, 101, 20, 45, 19, 122, 126, 49, -38, -37, 96, -36, 71, 18, -53, 14, 74, -41, -28, 31, -29, 126, 68, 29}
, {-31, 34, 33, 0, 34, 39, 36, -27, 48, 15, 56, 5, 21, 19, -68, 17, -6, -30, 31, -78, 35, 68, 40, 25, -18, 63, 11, 6, 29, 65, -45, -21, 3, 19, 12, 31, 29, -10, 44, -11, 50, 10, -1, -45, 74, 70, -11, 19, 29, -53, 9, -53, -13, 31, 62, -1, 31, -21, 15, 98, -40, -84, 33, 0}
, {58, 36, 76, 47, 47, -5, 31, -33, 55, 154, 64, 142, 60, -42, -22, 18, -5, -52, 21, 25, 11, 39, 68, 94, -17, 88, 1, 73, 49, 62, -17, 8, -25, 47, 47, -17, -9, -46, 30, 21, 66, -15, -2, -32, 86, 124, 40, 39, 23, -78, 128, 66, 41, 28, -19, 27, -27, 105, 12, 138, -28, 39, 89, 29}
}
, {{-95, -51, -61, -4, 35, -32, 54, 66, -56, 26, 40, -35, -19, 9, 35, -1, 50, -5, 6, 66, 46, -12, -56, 9, 18, -60, 47, 2, 65, 49, 11, 47, 60, -58, -69, 60, 72, 31, -30, 9, 68, 81, 19, -54, -90, 30, -40, 41, 50, 13, 44, 45, 13, -22, 0, 107, 48, 33, -72, -58, 124, -47, 55, -12}
, {-80, 46, -26, 0, 56, -4, -26, -75, -40, 39, -4, 4, 56, 59, -65, -39, 21, -24, 51, 0, 5, -46, 29, 74, -20, -4, 12, 42, -23, 27, -3, 118, 35, -28, 7, 1, 83, -14, 138, -14, 59, 30, 79, -10, -12, 33, -37, -65, 73, 87, 29, 46, 28, 50, 76, 40, 69, -96, 115, 13, 106, -1, 24, 76}
, {-2, -98, -25, 40, 66, 4, 21, -33, -52, -95, -22, -89, 63, -15, -105, -44, -49, -92, -50, 68, 23, 35, -7, 54, -6, -92, 56, 83, 122, 74, -97, -23, 87, 82, -5, -46, -56, -40, -41, -13, 28, 58, -44, -4, 7, 98, 56, -8, 88, 16, 23, 11, -40, 58, 27, 75, -17, 26, 68, 33, 152, 5, -61, 82}
}
, {{-44, 36, -32, -25, 84, 13, -10, -117, -26, -16, -14, -168, 40, 140, -97, -25, 35, -100, 26, -128, 35, -86, 23, 74, -22, -42, 76, -25, 15, -30, -13, 117, 102, -44, -91, 17, -12, 106, 73, -64, 133, -56, -22, -22, -23, -52, -9, -55, 29, -129, -32, 50, -45, 29, 63, 80, -1, -88, 50, 75, 24, -4, 29, 97}
, {7, 3, 41, -22, -37, 59, 95, 26, 18, 40, -13, 4, 13, 86, 47, -22, 33, -90, 9, -4, 13, 23, -34, -2, -23, 60, 13, 30, 30, 58, -35, 77, 8, 55, 27, 45, 9, 30, 53, 14, 21, 44, 48, -16, 22, -61, 134, 36, 65, -68, 49, 46, 37, 108, 58, 48, 32, -1, -33, -24, -89, 13, -37, -54}
, {3, 76, 105, -12, 103, -38, 24, 70, 3, 59, 86, 3, 44, 65, 5, -44, 64, -79, 41, -2, -46, -27, -49, 56, -7, 111, 22, 20, 70, 37, 59, 41, 14, -41, -10, 51, 29, 4, 26, 17, 35, 109, 19, -6, 24, -14, 50, -65, 26, -66, 101, 102, 27, 105, -68, 71, 38, 23, 26, -47, 52, 18, -99, -9}
}
, {{202, 18, 5, -66, 86, -52, -43, 43, 21, 15, -81, 42, -118, 67, -126, 21, 4, 2, 74, 22, -36, 32, 48, 7, 43, 32, -14, 92, 14, -67, 58, -61, -12, 42, -72, -78, -39, 18, 79, -77, 108, -86, -18, 8, -24, -131, -24, -2, -53, 27, 13, -7, 26, 66, 22, -58, 93, 20, -4, 11, 12, 276, 19, 46}
, {-68, -23, 66, 4, 36, -44, 13, 33, 30, 1, -88, 43, 61, 68, 53, 96, -16, -46, 53, -19, 21, -134, 60, -34, 28, 88, -13, 11, 11, -7, 30, -132, -32, -74, 18, -29, -14, 72, 61, -39, -59, -57, 108, -34, -76, -76, -37, 119, -67, -155, 32, 5, -14, -89, -30, -59, -19, 66, -123, -8, -91, 79, -67, 63}
, {-11, 53, 113, 124, 17, -20, -31, 43, -24, 67, 85, -30, -45, 23, 7, -55, 18, -69, -13, -65, -67, -133, -100, 117, -82, 101, 12, 90, -115, -60, 33, -6, 64, -35, -121, -67, 218, 16, 49, -28, 18, 106, 133, -45, 100, -72, -17, 52, -87, 86, 55, 21, 50, 9, 23, -11, -2, 48, -12, 104, -18, 75, 10, 8}
}
, {{155, 24, 69, -5, -86, 37, -155, 155, 31, 35, 132, 121, -42, -115, -16, 65, 27, 37, 69, 64, 69, 126, -60, 75, -83, -7, 40, 33, 93, -98, 55, 4, -50, 52, 60, -26, 52, -65, 89, 134, 4, -17, -67, -20, 34, -96, -38, 110, -13, 80, 49, 87, 110, 56, -31, -150, -33, 137, 80, -85, 16, 133, 61, -101}
, {-17, -24, 17, -7, -27, -33, -27, 73, 53, 13, 88, 36, 3, 38, 67, 20, -11, -34, 1, -27, 49, -95, -61, 63, -38, -21, 21, -35, 18, -42, -65, -65, 47, -54, 16, 33, 38, 72, -20, 117, -200, 39, -51, -79, 2, 7, -53, 73, 5, -27, -24, 22, 21, 26, -13, 22, -96, -2, 39, 5, 39, 87, -98, 43}
, {-24, 23, -42, -17, -117, 70, -46, -5, 90, 39, -52, -68, -18, -149, 8, 26, -18, 1, 66, -54, -33, -2, -45, -46, 26, 39, 41, 4, 43, 124, -189, -37, -15, 34, 101, 105, 9, 50, -29, 141, -208, -9, -47, -56, -86, -47, 27, 16, 136, 40, 37, 0, 76, -136, 35, -6, -43, 29, 4, -24, 2, -29, -46, 13}
}
, {{-45, 20, 19, 19, -28, -51, 9, -29, 7, 73, -26, -46, 100, 60, 129, 75, -5, 26, 55, -110, 40, -69, -5, 6, 59, 57, -27, 37, 11, 39, 1, -155, 101, -33, -46, 0, 58, 46, -9, 35, -177, -53, 49, 21, -95, 54, -88, -1, 20, -122, -9, -25, 24, -5, 53, -43, -8, 42, -24, -8, -94, -157, -34, 66}
, {43, 13, 85, 65, -116, 30, -62, 57, -15, 42, 107, 8, 19, -33, -16, 48, 65, -33, 15, -22, 6, 21, -16, -41, 8, 40, 87, 82, 74, 72, -27, -12, -65, 58, 110, 40, 56, 130, -21, 131, -8, 63, -38, 54, 27, -3, 0, 81, 113, -37, 48, -8, 39, 68, 51, -26, -62, 88, 70, -33, -111, 51, 18, 39}
, {37, -17, -70, 28, -13, 30, -4, 54, 13, 13, 120, 12, 107, -16, 5, 28, 3, 9, -36, -22, 4, -25, -2, 71, 46, 29, 71, 45, 15, -72, -5, 44, 79, 31, -75, 113, 150, -1, -30, -33, 55, -64, 64, -78, -61, 22, -23, 24, 33, -32, 0, -15, -4, 91, -10, 138, 39, 55, 20, -1, 89, -45, -35, 93}
}
, {{-56, -105, -13, -62, -48, 87, 68, 56, -79, -13, 33, -63, -35, 60, 35, 1, -56, 21, 13, -11, 19, -7, -100, -64, 82, -58, -53, -53, -26, 136, -14, 46, 38, 61, 23, 54, 47, 17, -25, 49, 0, -71, 49, 9, -2, -27, -41, 64, 79, -42, 20, -53, 1, 9, 26, 58, 18, -13, -45, 17, 49, -94, 21, 36}
, {41, 119, -64, 16, 85, -30, 41, 85, 65, 39, 57, -118, 0, 38, -78, -27, 34, -70, 40, -8, -32, -56, -27, -47, -20, 33, -51, 138, 63, 75, 25, 90, 109, 94, -7, -22, 35, 16, 24, 58, -59, 84, 17, -44, -25, 21, -32, 11, -13, -14, -7, 51, -24, 100, -44, 4, -41, 42, 48, -63, -7, 50, 90, 13}
, {5, -39, -148, -45, 20, 10, 75, -54, 23, 58, 60, -36, 127, -11, -43, -37, 71, 21, 21, -97, 38, 26, -128, 42, 36, 5, -10, 11, 31, 82, -102, -25, -13, 72, 25, 0, 44, -19, 26, 77, 19, 8, 15, -8, 5, 6, 2, -25, 139, 14, 26, -16, 18, 38, 115, 66, -23, 63, 7, 1, 64, -88, 70, 73}
}
, {{5, 89, -160, 8, -47, 78, -83, 40, -169, -72, -44, 1, -105, -76, 21, -33, 72, -14, -14, 137, -119, 21, -51, -46, -32, -1, -66, -50, -8, 80, -37, 101, -46, -152, 144, 60, 87, -12, -52, -37, 99, -37, -95, -43, 0, -70, 59, -101, 16, -56, -16, -70, 6, 94, -84, 76, 54, -6, 86, -1, 86, -37, -53, -67}
, {-114, -12, -18, -154, -65, 75, 7, -18, -106, -29, 92, -74, 9, -7, -60, -24, 4, -36, -34, 13, 47, 23, -132, 100, -20, -57, 26, 15, -23, 92, 18, 25, 7, -67, -1, -42, 29, -6, 12, -53, -6, -40, -84, -71, -19, -32, 30, -68, -43, -60, -7, -1, 8, 58, 31, 75, 98, 46, 84, 55, 18, -57, 13, 26}
, {-50, 81, 32, 5, 28, -12, 25, 97, 66, 41, 97, 12, -51, 40, -48, 0, 52, 29, 11, -69, 68, -17, 68, 107, 83, -43, 33, 71, -38, -2, -7, 85, 4, 26, 25, 52, -76, -7, 55, 21, 177, 20, 0, -40, 41, -24, -66, -65, -7, -27, -81, -58, -19, 75, 62, -49, -1, 73, 45, -24, -92, 15, 93, -32}
}
, {{44, -82, -32, -45, 95, 36, 57, 129, 86, 3, 5, 33, 17, -34, -53, 128, -28, -30, -47, 27, -1, 42, 22, 44, -83, 77, -32, -69, -56, -74, -45, 4, -15, -103, -20, -3, -47, 14, -45, -53, 99, -32, -45, -107, 11, -25, 95, -79, -80, 5, -20, -58, 6, -27, -168, -55, -39, 96, 72, -59, -49, 95, -121, -65}
, {1, 46, -6, -11, -38, -19, -14, 26, -16, -107, -16, 22, -27, -120, -79, -24, -93, -11, -22, -14, -112, -32, -35, 10, 34, 165, 34, -72, 9, -29, 83, 9, 22, -11, -35, -1, 53, -42, 39, -34, 61, -89, -86, -44, -15, -43, 98, -83, -60, 51, 27, -69, -50, 13, -98, -13, -59, -39, -22, 44, -11, 16, 25, 32}
, {42, 52, 38, -28, -86, -63, -13, 20, 37, -26, -29, -7, -80, 78, 27, -15, -69, -79, 62, 17, -41, -71, -34, -91, -16, 135, -90, 31, -9, -31, -73, 43, -18, -21, 129, 37, -160, 27, -36, 42, -72, 2, -119, 27, -111, -53, -28, -24, 97, 36, 49, -39, -96, -66, 113, 13, -29, 74, -98, -103, -2, 19, 4, -68}
}
, {{17, 49, -19, 27, 107, 122, 56, -6, -37, -17, 47, 1, -8, 16, -30, -18, -70, -80, 6, 29, 35, 19, -19, 55, 8, 26, 15, 7, 60, -24, 0, 47, 103, 64, -47, 13, 7, -5, 65, 92, 141, -29, -12, -1, 54, 7, 74, -8, 24, 53, 99, 35, 58, 16, -1, 37, 101, 53, 58, 117, 63, 47, 32, 73}
, {-52, 31, 16, 26, -22, 13, 9, 34, -36, 12, 75, 4, 41, 39, -107, 78, 76, -15, 55, 4, 6, 63, 31, 32, -12, 97, -8, 12, 52, -43, 47, 55, -68, -61, 49, 79, 72, -15, 50, 34, -25, 58, -7, -59, -99, 15, 78, 19, 59, -32, 74, 107, 42, 75, 47, 5, -1, 44, -4, 105, 6, 120, 46, 57}
, {47, 9, -15, -30, -7, 10, 39, -3, 29, 10, 27, 19, 8, 87, -120, 108, 101, 40, 98, 22, -19, 84, -25, 92, 55, 22, 14, -60, -6, 106, -39, 191, -53, -109, 22, 13, 103, 53, -1, 50, 83, 69, 6, -39, -20, 21, 107, -21, 45, -174, 86, 57, 6, 78, 13, 117, 12, 82, 93, 53, 49, 36, -15, 56}
}
, {{-40, 48, 7, 131, 49, -82, 10, -10, 72, 95, 67, 7, -35, -9, 67, -18, -5, -57, -27, -77, 5, -105, -50, -33, 59, -35, 39, 26, -24, 2, -35, -127, 97, 0, -57, -69, 105, -17, -64, 28, -260, 37, 8, 32, -38, 62, 5, 84, 21, -6, 95, 29, 21, 14, -5, -87, -67, 47, -36, 37, -33, -55, 86, 25}
, {-11, -23, -30, 95, 9, -11, -68, 19, 14, 70, 63, -45, -19, 45, 28, 29, 34, -119, 6, -46, -8, 2, 47, -17, 18, -53, 76, 28, -62, 13, -50, 9, 105, 14, -38, -44, -18, 54, -55, 82, -27, 61, 89, 26, -54, 9, 4, 134, 42, -40, -11, 17, 26, 34, 1, -2, 49, -29, -46, 60, 57, -39, 62, 25}
, {3, -26, -17, 95, 17, -1, -28, -55, 70, 104, 75, 28, -105, -55, 6, -27, 28, -1, -58, -86, -18, -9, 53, 57, -50, -79, -34, 37, -38, 32, 40, 37, 186, 19, -45, 126, 69, -26, -63, 79, 83, 78, 15, -24, -32, 66, 2, 115, 20, 30, 39, 0, 3, 111, 35, 94, 35, -23, -8, 116, 148, 120, 223, 15}
}
, {{-103, 56, 67, -21, 67, 5, -71, 40, -29, 7, 5, 55, 111, 0, -120, -18, 11, -80, 64, -56, 13, 105, -43, 53, -81, 44, 69, 83, 71, -1, -71, 112, 88, 47, 24, 74, 61, 10, -27, 63, 49, 97, 157, -39, -100, 37, 47, 43, 27, -1, 11, 72, 80, 22, -74, 2, 11, 50, -6, 69, 113, 162, 97, -89}
, {-151, 34, -24, -18, -31, 39, 18, -130, 13, 23, 23, 33, 74, -81, -82, -8, 57, -40, 35, -55, 72, 89, -99, 11, -29, 1, 44, -19, 45, 61, -101, 37, 36, -3, -20, 7, 93, -10, 41, -37, 50, 42, 95, 53, -35, -36, -32, 83, -21, -83, 46, -9, -28, 19, 68, 68, 51, -15, -18, 74, 82, -57, -11, -5}
, {-144, 39, 1, 29, 29, 15, 3, -106, -71, 64, -45, -8, -52, 6, -56, -48, 139, -32, 50, -91, 25, 74, -20, -19, 29, 26, 40, 20, 93, -65, 40, 110, -50, 87, -109, -4, 67, -54, -74, -20, -47, 97, -1, -59, 163, -24, 72, -1, 79, 7, 20, 12, 59, -1, -65, 52, 25, 18, 49, 34, 130, 24, -30, -47}
}
, {{45, -43, -61, 8, -98, -92, 13, -31, -18, 75, 45, 92, -13, -84, 45, 43, -54, 53, -20, -6, -16, -5, -50, -19, -162, 10, 4, -85, -58, -70, 83, 31, 1, -29, -24, 90, -10, -26, -111, -19, -26, 46, -96, -106, 46, 89, -7, -43, -74, -16, 17, -3, 42, 21, -118, -20, -1, -7, 56, 12, 43, 127, -74, -14}
, {23, 54, -49, -3, -46, 68, -38, -8, -38, 27, 32, 102, 11, -85, -69, 16, 9, -13, 18, -10, 12, 13, -100, -42, -24, -5, 4, 35, -49, -50, 19, -23, -39, -78, -35, 60, -5, -149, -4, 69, -26, -9, -44, -76, 7, 31, 32, 26, 46, -18, 125, -35, 0, 37, -101, -22, 15, 56, 63, 9, -15, 18, -33, -94}
, {30, 11, -55, 67, 41, 12, -7, -47, -81, 18, 23, 181, -89, 46, -98, 38, -41, 15, 36, 5, -83, 18, 97, 40, -93, -50, -62, -51, -21, -116, -8, 1, -62, -11, -49, -106, 22, -182, 70, -33, -29, 59, 2, -20, -37, 68, -41, -112, -98, 155, 22, -70, -51, -45, -117, 82, -86, -88, -15, 23, -43, 91, -205, -72}
}
, {{-25, -92, 159, 15, -80, -76, -72, -30, 127, -56, 17, 35, -38, -115, 174, -10, 73, 46, -46, -224, -144, -143, -13, -46, 43, -112, -6, -23, -56, -34, -78, -123, 17, 120, -25, 73, 25, 40, -143, 39, -232, 63, 13, 19, -164, 74, -115, 322, 97, -65, 74, 68, 38, 37, -121, 6, -58, -92, 38, -25, -45, -25, 20, 32}
, {108, -8, -160, -47, 18, 12, 34, 10, -35, -57, 53, -17, -73, 39, -10, -14, 22, 87, -1, -37, -86, 19, -148, -33, -58, 61, -92, -162, 0, 23, -82, 31, 21, -124, -86, 115, 74, 33, -51, -73, -2, -76, -4, 93, -53, 26, -22, 147, -24, 7, 0, -22, -28, -57, 47, 37, 21, -40, 19, -95, -11, -36, -28, 41}
, {39, 61, 7, 33, 75, -54, 208, 60, 91, -85, 20, -66, 8, 25, 44, 23, -28, -64, -3, -4, -16, -52, -17, -11, -27, 43, 33, 92, -29, -23, 5, 6, -34, -51, -79, 24, 15, -70, 47, -8, -33, -103, 0, -9, 37, 34, -64, -102, -54, -28, 52, 22, -107, -102, 27, -33, -82, 221, -9, -62, -31, -38, -10, -7}
}
, {{-67, 91, 51, 67, 29, 54, -54, 8, -4, -15, 41, 7, 142, -75, 11, 7, 35, 27, -65, -52, 6, 45, 43, -31, -29, 58, 222, -25, -30, -18, 20, 45, 103, 121, 110, 111, 47, 54, -34, 162, 15, -6, -64, -118, -30, 133, -93, 84, 72, -39, 191, 86, 39, 27, -14, 33, 12, 89, 32, 56, 29, 1, 41, -16}
, {-40, 47, -78, 15, -34, 73, 71, 7, -71, 24, 0, 84, 32, 35, 32, 74, 45, 11, -17, -17, -37, 79, -57, -27, -85, 20, 119, -22, 68, 30, 18, 57, 44, -7, 34, 41, 106, 51, -23, 89, 4, 69, -14, 18, -13, 121, -27, -103, -2, -54, 102, 58, 14, 80, 52, 24, 131, -48, 2, -23, 70, -62, 39, 50}
, {-44, 64, -73, -18, -14, -56, 110, -1, 45, 29, 25, -9, -99, 85, -79, 83, 3, -31, -7, 19, -11, 42, -34, 53, 15, -72, 24, -14, 26, 45, -35, 36, 45, 67, -19, 81, 56, 50, -15, 8, -11, 24, 56, 49, -3, 3, 118, -24, -4, -70, 13, -30, 1, 86, -49, 38, 40, 6, -54, 44, 44, -11, -5, 100}
}
, {{80, -17, -55, -39, -88, 103, -47, 79, -111, -17, 101, 119, 16, -110, -111, 52, 41, -60, 50, -9, -31, 117, -92, 49, -65, -17, 7, 95, -9, 29, -34, 93, 5, -1, 53, 120, 64, -38, 25, 45, 10, 28, -66, -73, 6, 37, 126, -6, 115, 61, 116, -97, 85, 45, 34, 43, -31, -85, 54, 72, 79, 50, 154, -118}
, {-44, -4, -31, -88, -13, 17, 1, -115, 45, -25, 19, -4, 9, -43, -19, 36, 19, -1, 51, -9, 8, 46, -65, 57, -48, 15, -1, 8, 63, 46, -67, 106, 38, -27, 35, 44, -52, 4, 43, -19, -1, 27, -7, -48, -47, 76, 44, -69, 95, 14, -68, -36, 69, -15, -35, 93, 32, 12, 31, -69, 34, 57, -51, -5}
, {-67, 35, 32, -16, -17, -40, 23, -34, 87, -31, -39, 48, 97, -28, -22, 33, 49, -76, 72, -27, 102, -46, -13, 90, -32, 19, -7, 20, -1, 16, -78, 14, -29, 46, 45, 10, -62, 73, 69, 83, 7, -22, 5, -79, 25, 8, -32, -68, 104, -16, 0, 72, -70, 81, 26, -103, -18, 34, 75, 40, -49, 60, 18, -3}
}
, {{3, -68, 154, -52, -110, 29, -4, 98, 119, 45, 27, 48, -52, -46, -3, 3, -39, 64, -51, -52, 64, -57, -14, 8, -104, -57, 31, 19, -74, 1, -31, -162, -106, -10, 43, 60, 9, -69, 98, 129, -192, 102, 22, 6, 45, 6, -97, 46, -3, -48, 37, 19, 73, 56, -47, -94, -88, 42, -102, -87, -104, 1, -14, -44}
, {-5, 57, 40, -30, -4, 1, 10, -26, 48, 51, 1, 31, 30, -17, 90, 60, -51, -48, -66, -35, -54, 51, 52, -9, -38, -68, 30, -53, -55, -54, 17, -12, -20, -27, 83, -118, -93, 39, 34, 9, 75, 50, -124, -32, 90, 1, 13, 16, -41, 62, -65, -98, 11, -3, -34, -158, 19, -51, 65, -18, -35, 63, 13, -75}
, {35, 23, 65, -21, -53, 20, -118, 6, 71, 84, 13, 44, 81, -35, 93, 12, -3, 7, -23, -50, 58, -55, 7, -24, -65, 9, 32, -54, 21, 4, 144, -109, -99, 7, 98, 38, -50, 62, -54, 47, -26, 27, -97, 75, 34, -29, 18, 8, -33, -87, 69, -101, 33, 13, -10, -337, 24, 7, 3, -92, -162, 88, -128, -29}
}
, {{-28, 29, 89, -125, -3, -72, 99, 71, 62, -59, 189, 8, -20, -1, 19, 85, 27, 88, -31, 90, 42, -6, -64, 42, -104, -42, -51, -80, -69, -102, 72, -6, -69, -144, -61, 30, 13, 11, -58, -72, -22, -32, -20, 2, -58, 93, 25, 28, -63, 65, 145, 48, 74, 0, -178, 36, -15, 79, -137, -90, 77, -33, -2, -100}
, {14, 1, -82, -15, 5, -12, 21, -94, -38, -15, -7, -101, -8, 14, -74, 0, -36, -6, -120, 87, -95, -55, -78, -28, -92, -7, -31, -7, -30, -27, 36, -66, -3, -99, -64, 19, 157, 51, -94, -23, 60, -24, -46, -27, -20, 16, -57, -5, -79, 2, 1, -16, -31, -11, -43, 15, 85, 14, -69, 46, 25, 23, -11, 7}
, {-107, 44, 120, 40, 20, 26, -91, 13, 145, 40, -92, 77, 34, -31, -108, -52, -262, -88, 153, 29, -2, 156, 178, 113, -14, 214, 75, 70, -86, 6, -67, -81, -71, 49, 20, 91, -179, 116, 35, 42, -173, -90, -7, 49, -155, 12, 37, 122, 66, 35, 168, 24, -40, -213, 82, -198, -124, 210, -15, 8, -127, -47, -70, 16}
}
, {{20, 33, -3, 31, 10, 15, 48, -7, 28, 61, 76, -70, 85, -16, -84, -120, 7, -72, 84, -25, -3, 4, -87, 71, 18, -3, 25, 119, 29, 57, -26, -15, 33, 13, -48, 35, 13, 2, 110, -27, 13, 2, 57, 67, 74, 108, 22, -55, 40, -42, 21, 49, 81, 57, 58, 18, 59, 15, -29, -57, -85, -19, 142, -40}
, {66, 11, 13, 24, 3, -70, 22, 13, -122, 57, 25, 27, -75, 50, -49, 3, 58, -56, -20, -16, 13, -26, 27, 6, 43, 40, -1, 67, 17, 10, 98, 72, 68, 17, 34, -103, 46, 7, 16, 11, 63, 59, 14, 26, 29, 1, -23, -35, 21, 93, 67, -47, 27, 35, 11, -79, -31, -129, -5, -15, -2, 46, 94, -12}
, {14, 58, 27, -52, 38, 91, -70, 30, 17, 108, 106, 80, 60, -45, 33, 10, 86, 98, -17, -145, 20, -2, 65, 15, -34, 76, 43, 17, 145, 19, -29, -49, -5, 6, -44, -3, 40, -62, 161, 10, 10, -54, 86, -21, 55, 19, 49, -68, 56, -45, 37, 54, 39, 51, 81, -78, 170, -48, 42, -47, -13, -8, 88, 144}
}
, {{-2, 141, 53, -79, 32, 93, 36, 67, -96, -19, 77, 39, -36, -57, -134, 35, 83, -99, 46, 110, 22, 106, -98, 73, -150, 51, -42, 85, 37, 28, 0, 69, -111, -39, 63, 8, -59, -54, -16, -29, 134, 11, -64, -52, -24, -54, 119, -78, -66, -12, 23, 24, 112, 117, -10, 58, 28, 80, 150, -46, -20, 95, 25, -93}
, {-55, -54, 110, -52, -3, 15, -22, 69, 44, -17, 34, 32, 93, -61, 53, 89, 22, 9, 20, 11, 81, 80, 27, -1, 8, 13, 33, 15, 39, 61, 12, -62, -73, 33, 35, -3, -116, -22, 3, 7, 12, -22, -47, -101, 81, 19, 55, -30, -12, 59, 31, -10, -30, -69, -10, -6, 2, 117, -31, -26, -42, -48, -2, -26}
, {-80, -18, -9, -54, -63, -12, -2, -52, -108, -99, -13, 34, -77, -30, 53, -57, 99, 6, -165, -45, -93, -95, -113, -86, -60, -21, 33, -75, -8, -4, 38, 126, 193, -66, -62, 67, 3, -61, -143, -25, 7, 17, -93, -23, -82, 30, -32, -2, -40, 142, -55, -153, -29, 51, -140, 157, -8, 15, -76, -13, 148, 50, 67, 0}
}
, {{169, 84, -79, -42, 26, 49, -99, 1, -38, -72, 52, 42, -89, -14, -49, -46, -66, 26, 55, 24, 18, -58, -16, 30, -16, -38, -9, 32, -61, -82, 69, 64, 30, -27, 8, -59, 16, -11, 10, 24, -43, -29, -14, -10, 82, -155, -114, -78, -26, 127, -34, 24, -38, -4, 18, -63, -52, 32, 7, -62, -42, -27, -28, -34}
, {-3, 7, 77, 10, 32, -20, -70, -20, 70, 36, 26, 5, -10, -87, 62, 38, 1, 90, 23, -15, 13, -14, -4, 24, -31, -82, 0, -33, 11, 10, 19, -102, -2, 10, 45, 63, 1, 55, -78, 7, -67, -27, 21, -17, 63, 32, -78, 69, -2, -76, -38, 16, 29, 33, 7, -41, 7, 50, -49, -69, 37, 2, -14, -75}
, {28, -69, -13, 36, -6, 4, 44, -57, -56, -52, -47, -1, -73, -46, 2, 69, 39, 23, -31, 4, -110, -18, -37, -13, -57, -116, 12, 10, -84, 28, 13, 60, 38, 1, -43, 15, 129, -52, -7, 29, -11, -10, -42, -33, -69, 23, -99, -7, -22, 76, 43, -67, 27, 59, -40, 54, -94, -94, 78, 13, 67, -2, 65, -96}
}
, {{-85, -91, -7, 66, 69, 8, 0, 2, 28, 3, 40, -31, -31, 57, -87, -40, -41, -85, 17, -116, 70, 15, 4, -68, -64, -9, 6, 68, 18, 17, -50, -80, 46, 9, 2, -7, -22, 7, -3, -2, -64, -15, 36, -31, 73, 78, 61, 15, 55, -126, -80, -55, -24, 96, 37, -55, 55, -6, 8, -55, -72, -41, -35, 43}
, {-40, 9, -49, -23, 18, 56, -74, -57, 64, 25, 64, 96, 89, 38, -12, 18, 100, -15, 62, -70, -32, 45, -72, 94, 17, 45, 53, 57, -3, 68, -12, 92, 33, 35, 108, 34, -8, 24, -1, -18, -22, 67, -66, -73, -46, 11, 9, 60, 1, -30, 37, -21, -9, 41, 32, 110, 54, 23, 93, 4, 101, 49, 47, 24}
, {-20, 88, 31, 5, -25, 114, -14, 8, 71, 73, 99, -18, 13, -16, -60, -27, 18, -26, -15, -62, 0, -76, -65, 46, -22, -18, 46, 61, 65, 20, -56, -4, 19, 41, -4, 41, 19, -43, 125, -89, -39, 45, 13, 46, -20, -1, 62, 66, -26, -53, 25, 43, 53, 105, 104, 67, 69, 8, 34, 4, -7, -37, 45, 65}
}
, {{-64, -26, 110, -101, -44, 119, -41, 117, 21, -71, 84, 170, -29, -88, -37, -84, 97, -24, -71, -51, 15, 119, -147, -52, -119, -53, 129, -33, -32, 65, -21, 60, -37, 86, 70, 112, -78, -86, -98, 57, -18, 24, -46, -82, -74, 131, 10, 111, 128, 75, 49, 125, 186, 41, -121, 81, 56, 34, -61, -82, 5, 154, 29, -97}
, {-42, -56, -9, -22, 40, 1, 52, -41, -20, -73, 15, 153, 49, 73, -80, -112, -75, 64, -101, -5, 14, -39, -45, -37, -92, 45, 21, -105, -73, 123, -47, 29, 71, -79, -25, -61, 56, -108, 1, -16, 55, -64, -18, -77, 3, 14, -58, -46, -56, 25, 38, 59, -35, 39, 23, 83, 90, 49, -71, 42, 45, 79, -135, 9}
, {142, -47, -57, 142, 17, -71, -20, 72, -1, 133, -5, 19, -72, 53, -13, -126, -70, -25, -38, 0, -43, -99, 53, 67, -68, 10, 37, 49, -198, -142, 141, 50, 8, -12, -175, -174, 78, -127, 47, -66, 52, 84, 51, -154, 29, -13, -34, 7, -160, 4, 65, -17, -101, -20, -110, 38, -136, -24, -69, 119, -14, 117, 65, 14}
}
, {{-7, 73, -44, -3, -28, -63, 27, -4, -179, -8, -61, 97, -8, -64, -97, 36, 12, 48, -84, 62, -23, -20, 23, -17, 11, 20, -13, 0, 59, 15, 144, 71, -1, -16, 29, 2, 10, -27, -63, -4, -6, -53, -124, -128, -30, -100, -65, -130, -62, -43, -2, 7, 13, -24, -126, 4, 28, 57, 33, 22, 127, 54, -30, -84}
, {29, -8, -2, -20, -27, 78, -22, -79, 7, 39, -74, -10, 23, 42, -14, -30, -23, 9, 80, -68, 67, -6, 47, 0, 15, -8, -56, -8, 5, -4, -75, -35, -102, -23, 40, -13, -21, 55, 13, -22, -77, -33, -12, -73, -24, -78, -23, -8, 33, 5, -12, 1, 13, -22, -27, -82, -72, 58, 42, 5, -53, -61, -26, -77}
, {-114, -56, -49, -57, -48, -26, 5, -13, -76, 63, -95, 89, -60, -32, -42, -94, -54, -10, 11, 133, 54, 98, 44, 29, -8, -152, -59, -55, -4, -22, 41, 87, 40, 0, -110, -131, 46, -125, 13, 26, -65, 19, -12, 8, -43, -17, -11, -25, -102, 48, 73, 27, -18, 43, 11, -67, -68, 31, -80, 115, 77, 68, -128, -45}
}
, {{-79, -41, -13, 3, -49, 29, 46, 16, 24, 99, 19, -6, 10, 6, 25, 38, 10, -58, -37, -8, 71, -81, 13, 10, -41, 26, 48, 23, 28, 74, 47, -99, 37, -40, -94, -110, 46, -13, -65, -9, 26, 19, 16, 31, 20, 30, 19, 18, -36, -29, 6, -28, -14, 58, -39, 127, -47, 22, 5, 42, 26, -77, 35, 77}
, {10, 23, -15, 25, -3, 26, 26, 69, -37, 68, 34, -6, 35, -10, -36, 33, -72, -45, 23, 28, 95, 15, 71, 76, -23, -20, 26, -37, 16, -89, 61, 70, 76, -98, -62, 28, 138, -28, -6, -34, 72, 69, 39, -79, 116, 55, 139, 4, 59, -22, 7, 18, -52, 6, -37, -22, 88, -84, -26, 75, 7, -13, 42, 10}
, {-75, -67, -134, -46, 28, 48, -97, -57, -112, -61, 43, -4, -14, 13, -105, -38, 63, 69, -119, -33, -102, -19, 104, -82, 34, -131, 5, 31, 29, 31, -70, 51, 127, 26, 38, 141, 89, -132, 34, 38, 109, -46, -1, -18, -88, 1, 35, 62, -3, -31, 19, -60, 17, 105, -19, 38, 113, -153, 46, 79, 41, 19, -22, 1}
}
, {{28, 82, 111, -6, -75, 0, -70, 81, 24, 1, -16, 40, 33, -123, 0, -35, -31, -49, -37, 22, -3, 27, 17, 94, 30, -10, 120, -50, -29, -2, -48, -65, 79, 139, 104, 35, 57, -19, -64, 211, -82, 5, -34, -19, -65, 61, -6, 24, 2, -5, 161, -11, 29, 29, -11, 13, -82, -51, 65, 57, -55, -23, -7, 32}
, {-59, 38, 11, -52, -16, 42, -69, 49, 6, -63, 50, 86, 5, 12, -24, -17, 47, -4, -27, 7, 28, 15, -68, 48, -45, 7, 100, -57, 0, 39, -53, 8, 35, -8, 35, 37, 19, -19, -17, 33, -12, -34, -22, -101, -63, -54, -73, -10, 22, -32, 31, 32, 29, 20, 13, 1, -8, 25, -20, -148, 10, -16, -68, -27}
, {-25, 39, 120, 12, 28, 99, 0, -19, -23, 55, -50, -10, 24, 97, 25, 72, 100, 36, -8, 14, 23, -50, -68, 59, -72, 22, 48, -5, -42, -26, -55, -2, 91, -39, -16, 86, 30, 33, 34, -10, -44, -44, 9, -42, -124, 46, -76, 22, 82, -13, 112, 53, 33, 22, 42, 97, 21, 75, 22, 3, 96, -45, 5, 115}
}
, {{-7, -6, -40, 45, -93, -100, 27, 52, -2, 42, 68, -47, 64, 31, 53, 106, 32, 61, 30, 112, 52, -60, -28, -54, 42, -26, -1, -10, 83, 35, 45, -8, -63, -33, -33, -57, 2, 7, -52, -12, -3, 14, 20, 16, -53, 66, -12, -19, -52, 31, -185, -33, -25, 60, 43, 15, -32, 44, -90, -102, -15, -31, -35, -75}
, {18, -98, 32, 29, 4, -57, 25, 66, -72, 3, 2, 0, 89, -35, 124, 78, -83, 39, 89, 95, 127, -64, 37, 76, 26, -136, 41, 42, -71, 46, -46, -20, 53, 21, 32, 1, 2, -34, -33, -132, -135, 19, -21, -136, 19, 62, -98, 11, 1, -16, 0, 56, -43, -45, -16, 24, -52, 4, 18, 7, 19, -6, 75, -40}
, {-21, -40, -245, -77, 142, 163, -109, -134, -305, -205, -65, -27, -25, 5, -158, -157, -87, 84, -150, 71, -84, 23, 145, -167, 87, -341, -141, -17, 81, 140, -12, 113, 89, 44, -19, 80, 85, -46, -58, 69, 194, -16, -48, -141, -94, -207, 19, -79, 55, 44, -77, -113, 104, 80, -16, 254, 37, -326, 174, 142, 96, -81, 63, 171}
}
, {{101, 44, -34, -19, -32, -96, -12, 43, -2, -40, 40, 97, 72, -98, -80, 63, -128, -17, 24, 40, 8, 33, 53, 96, -82, 47, -9, -110, -43, -50, 51, -12, 23, -37, 55, -59, -36, -45, -31, 67, 12, 55, -7, -57, -102, -30, 23, -161, 60, 20, 22, -35, 2, -49, -97, -112, -53, 22, 35, -20, 22, 28, -44, -86}
, {-2, -26, 15, 15, 23, -37, -5, 20, 11, -1, -8, -24, 30, -24, -53, -12, -20, 23, 30, 13, -57, 73, 71, 43, -56, 14, 10, -34, -56, -57, -120, -47, -67, 13, 34, 9, 53, -69, 24, 15, 55, 61, -51, -56, 26, 41, 5, -22, 6, 40, -27, -73, 65, 12, -2, -28, 24, 77, -12, 76, -64, 41, -12, -16}
, {-13, -19, -22, 29, -70, -35, 29, 10, 67, -72, -19, -38, 7, -34, -47, 33, -96, -88, 80, 83, -54, 81, 41, -21, 7, -53, -64, 9, -4, -25, 46, 9, -166, -78, 46, 22, -8, -4, 1, -2, 32, -69, -85, -83, 28, -21, 48, -72, -9, -39, 5, -93, -67, -93, -109, -60, -9, -43, 4, 25, 14, -2, -141, -55}
}
, {{-48, 31, -45, -9, 9, 161, -33, -29, -56, 38, 0, 18, 16, 5, -68, -83, 16, 83, 61, -98, 99, -2, -8, 85, 58, -53, -5, 80, -14, 169, -24, -12, -89, -87, 86, -64, 19, 14, 95, 43, 46, -20, 40, 48, 55, -102, 123, 45, -1, -118, -7, -72, -36, 34, 114, 25, -2, -7, 183, -41, -5, -32, 56, 18}
, {-22, 21, 93, -60, -48, 153, 53, 57, 36, 47, -8, 57, 144, 6, -65, 121, 5, -74, 44, -83, 114, 80, -44, 120, -57, -42, 58, -78, -84, 26, -53, -29, -63, 44, 59, 101, -5, 61, 28, 83, 19, 138, -14, 2, -22, 110, 37, 64, 44, -40, -37, -89, 9, 14, 20, -87, -6, 91, 66, 32, -34, -38, -26, -6}
, {-63, -7, 30, 39, -45, 51, 34, -2, -32, -9, 94, 18, 53, 40, -23, 148, 79, 75, -10, -130, 87, -20, -169, 37, -59, 32, 1, -52, -31, 2, -45, 47, -25, 25, -31, -33, 66, 11, -9, -26, -111, 10, -65, -90, -15, 82, -47, 15, -46, -28, -93, -102, 38, 66, 69, 37, -14, 58, -22, -88, -83, 16, -28, -26}
}
, {{27, 36, 21, 23, 56, 97, -22, -30, -33, 18, 33, 54, -28, 47, -25, 114, 93, 19, -12, 8, -17, -20, 3, -1, 0, -24, 18, 5, 71, 13, -18, 14, 40, 32, 53, 70, 52, -2, -10, 4, -12, 79, -50, -108, -5, -35, 27, 17, 83, -12, -36, -51, -10, 80, -3, 64, 113, -4, 89, 34, 63, 17, 4, -29}
, {-4, 29, -38, -40, -57, 80, 58, -47, -65, 80, 65, -31, 47, 39, 46, 44, 76, -7, 11, -33, -67, 28, -8, -50, -86, -13, -48, 93, 30, 109, 54, 15, 30, 59, 129, 108, -27, 60, 8, 81, -48, 24, -59, 58, -53, 32, -31, 55, 60, -34, 40, 17, -24, 66, 66, 31, 9, 37, 21, 12, 64, -17, 40, 63}
, {-9, 18, -6, 23, 42, 23, 56, -2, -4, 103, 37, 77, -13, 80, -9, 50, 32, 30, -54, -94, -6, 46, -41, 18, -55, -3, -25, 28, 1, 45, 3, 59, 53, 20, 38, 102, -42, -12, -44, 8, -23, 69, 14, 49, -65, 60, 9, 39, 38, -51, 84, -34, -24, 100, -12, 131, 26, 59, -46, -9, 31, 17, 49, 34}
}
, {{40, 94, -16, 8, 35, -32, 60, 110, 103, 38, 85, 51, 32, 100, -109, 69, 10, -56, 37, -66, -43, 10, -45, 19, 19, 55, -7, 74, 20, -36, -44, 65, 63, 50, -42, -50, -11, -3, 215, 38, 48, 54, 40, 7, -70, -21, -3, -88, -17, -117, 103, 159, 28, 54, 70, -5, 1, 69, 151, -78, -40, -5, -45, 86}
, {40, 10, 45, 76, 76, 73, 34, 81, 50, 90, 81, 0, -69, 6, -6, -24, 11, -66, -39, -40, 55, 32, -55, 82, -28, 32, 112, -41, 60, 62, -52, 47, 57, 19, 4, 103, -6, 55, 19, 83, -60, 62, 118, -8, 19, 85, 74, 173, -7, -167, 87, 18, 20, 102, 36, -35, 35, 30, 89, 21, -38, 40, 61, 12}
, {-49, 29, 101, 27, -54, -27, 72, 45, 12, -60, -37, 26, 72, 61, -15, 36, -11, -9, 5, -77, 70, -53, -20, -12, 51, 64, 43, 34, -21, 21, -89, -69, 106, 11, 6, -33, 47, 33, -32, 79, -15, 95, -66, -9, -82, 4, 149, 46, -13, -121, 11, 51, 13, 38, -52, -24, 49, 105, -17, -2, -144, -102, -107, -7}
}
, {{-63, 34, -102, 6, 40, 13, 34, -73, -82, -59, -23, -75, 64, 71, 41, 70, 30, -100, -67, -53, 28, -29, -101, 0, 38, 6, 4, -73, 19, 28, -27, 119, 11, 18, -9, 37, 11, 58, -27, -49, -29, 6, 99, -76, -33, 63, 17, 9, 63, -25, 47, 9, 33, 60, 53, -3, -60, -10, 25, -17, 6, -84, -62, 88}
, {-90, 58, 12, -70, 47, 15, 125, 37, 26, 36, 27, -96, -51, 3, 32, 20, 112, -55, 10, -5, -71, 15, -76, 89, -18, -63, -2, 50, 26, -13, -43, 65, 56, -14, 4, 1, 16, -34, 62, -6, 82, -44, -36, -43, -96, 21, -30, -104, 19, -96, 16, 90, 79, 83, 3, 61, -53, -14, 28, 87, -38, 10, 6, 41}
, {-53, 85, -56, -93, -11, 28, 76, -35, -54, -54, 41, -103, 36, -34, -54, 44, 25, -78, -9, -50, -38, 68, -65, -47, 20, 34, -20, -92, 107, 65, -28, 29, 17, 0, 63, 78, 68, 62, 23, 116, 43, -12, 59, -74, -30, -65, 14, -92, 55, 27, 20, 38, -29, 27, 37, 78, -34, -66, 16, -15, 43, -53, -74, 47}
}
, {{-51, -48, -103, -21, -37, 210, -82, 119, -140, -13, 5, 30, -97, 28, -27, 40, 19, 178, 113, 28, -40, -46, -38, -19, 3, -106, -19, 4, -55, 12, -77, 45, 7, -124, 120, 213, 100, -68, -15, 8, -35, 67, 7, -45, -103, -30, 59, 83, 27, 30, -15, -174, 133, 76, -82, 137, -44, -139, -39, -164, 42, 135, -58, -11}
, {9, -21, -43, 77, -70, 90, -56, -49, -58, 41, -30, 18, 8, -19, -13, 16, -17, 56, -55, 19, 85, 16, 93, -122, -14, 21, -96, 57, -57, 17, 32, -36, -37, 4, 32, 50, 42, -14, 26, -25, 18, -64, -95, -32, -26, -139, -39, 24, 9, -15, 68, -85, -70, 48, 7, 7, -20, -82, 41, -51, 25, 7, -19, -33}
, {25, -2, -66, 0, -158, 94, -45, -93, 36, 70, -14, 125, 65, -41, 19, 60, -93, 73, 12, 46, 41, 4, -26, -42, -13, -97, -41, -178, -27, 49, -3, -55, -164, -6, 167, 5, -114, 79, -128, -44, -60, -105, -128, -23, 15, -19, -6, 29, -54, -28, 24, -71, -35, -39, -2, -38, -53, 58, -38, -130, -8, 68, -146, -44}
}
, {{-13, 8, 101, 44, 41, 27, 19, 27, -13, 27, 29, 74, 18, 92, -51, 114, 35, -53, 28, 12, 28, -66, 43, 72, 45, 117, 78, -65, 29, -33, 31, 147, 78, 87, 50, 4, 56, -21, 21, 53, -7, 53, 56, -52, -61, 66, -7, 68, 61, 10, 66, 61, 0, 26, -69, 0, -133, 81, 51, 103, 64, 96, -32, 71}
, {-9, 63, -2, 50, 41, 37, 4, 27, 97, -17, 43, -47, -3, 98, 8, -56, -11, -79, -24, -40, 7, -69, 33, -12, 38, 63, 21, 30, 65, 61, -68, 25, 138, 24, -26, -6, -10, 78, 58, 94, -73, -16, 40, -1, -40, 53, -18, 4, 41, -35, 42, 31, 19, -19, 67, 64, -48, 88, 20, 64, 87, 38, 23, 90}
, {-79, 48, 72, 94, 52, 87, 13, 32, 147, 38, 22, -13, 126, -36, -6, 21, 8, -66, 39, -16, 120, -46, 96, 72, 58, 15, 1, 7, 34, 70, -73, 137, 140, 31, 92, 81, -87, 139, 23, 135, -89, -20, 23, -79, -22, 56, 31, -5, 35, -64, 85, 14, 51, 36, 83, -24, -98, 145, 86, 84, 69, -16, 23, -12}
}
, {{58, -49, -69, 30, 33, -75, -29, 100, -14, -41, 71, 8, -29, -24, -95, 43, -5, -88, 59, 44, 31, -60, 157, -25, -18, 16, -36, -109, 12, -47, 17, 7, 6, -3, -39, 12, 17, 16, -3, 22, -120, 87, -38, 21, -102, -22, -27, -42, 84, 43, -81, -28, -17, -10, 14, -3, -48, 17, -96, -116, -99, 141, -76, -37}
, {25, -56, -58, -1, 3, -154, 72, -76, 27, -28, 29, 0, 24, -40, 19, -21, -1, -14, 32, 82, -10, -47, 60, 42, -7, -20, -35, -75, 13, 14, 42, -4, 66, -37, -13, 6, 93, 28, 21, -61, -68, -90, 40, -23, -34, 33, -59, 7, -34, 8, -15, -83, -18, -64, 123, -78, -26, 60, -91, 32, -71, -132, -6, -58}
, {53, -47, -14, 28, 77, -56, -38, -47, 0, 94, 74, 85, -4, -17, 60, -69, -99, 129, -55, -27, -30, -64, 91, 17, -16, -28, -10, 67, -102, -60, 19, -16, 97, -10, -91, -2, 24, -34, -87, -79, -105, -70, 2, -18, -16, -4, -113, 136, -149, 30, -84, -49, -57, -29, 57, -48, -18, 75, -89, 37, -48, 6, 44, 49}
}
, {{-88, -1, -40, 11, 5, 35, -12, -51, -101, -32, 74, -5, -63, 42, -101, 44, 50, 4, 18, 8, -25, 17, 3, 75, 22, -7, -33, -42, -54, -121, -69, 41, -57, -61, -12, -20, 42, 81, 64, -48, -71, -18, -10, -42, 0, -59, -6, 48, 7, -9, -69, 16, -52, -56, -41, 2, -7, -114, 55, -86, 15, 20, -65, -11}
, {-60, 31, 9, -96, 39, 32, 38, -67, -54, 19, 71, 0, 57, 29, -39, -90, 37, -48, -23, -79, -4, -11, -24, 59, 20, -53, -19, -26, 38, -9, -3, 101, 3, -37, 15, 28, -24, -6, 61, 52, -24, 16, -24, -34, 23, -7, -51, -44, 11, -35, 109, 58, 74, -22, 23, -5, -46, -17, 30, 7, -59, -58, 10, -69}
, {-27, 98, 125, 41, 49, -14, 116, -53, 24, 95, 92, 15, 41, 57, -11, -59, 63, -61, -54, 11, -5, 5, 31, 93, -103, 146, 131, 131, 17, -17, -41, 33, 37, 24, 28, -1, 67, 5, 25, 108, -18, 88, 192, -74, 56, 106, 14, -11, 26, -127, 158, 14, 87, -2, 63, -40, 26, 168, 27, 117, 11, -30, -9, 51}
}
, {{-54, 53, 76, 26, -34, 13, -25, -17, 78, -78, -75, -104, 31, 69, 158, -34, -31, 16, -40, -120, -130, -119, -6, -70, -71, 3, -94, 18, -34, 9, 14, 8, 62, 0, -51, 5, -49, 2, -21, 3, 13, 0, -40, -27, 76, 42, -99, 74, -58, -8, -66, -60, -57, -68, -86, 16, -17, -64, -2, 4, -67, -159, -50, -11}
, {11, -36, -9, -43, -62, 67, -24, -15, -73, 34, 47, -65, 2, -72, -45, -28, -3, 85, 19, -28, -112, -71, 35, 30, -34, 65, 4, -26, -27, -61, 31, 6, 32, 25, -24, -29, -12, -21, 49, 45, 110, 9, -85, -43, -51, -26, -45, -16, 15, 99, 72, -49, -76, -61, -46, 67, -4, -43, 31, -4, 111, -48, -57, -25}
, {-23, -31, 42, -3, 17, -48, 39, 24, 39, -39, -32, -59, -37, -47, 49, -86, -81, -1, -5, 14, 17, -31, 50, -77, -11, 127, -13, -67, -68, -32, 32, -5, -10, -60, -65, 27, -49, -44, 90, -22, 45, -45, -44, -97, 32, 19, -87, -55, 35, 19, -93, 39, -31, 41, 2, -39, -15, 97, -18, 14, -6, -43, -6, 1}
}
, {{32, -55, -282, -11, 13, 286, -105, -124, -237, -195, -131, -3, -7, -44, -122, 50, -6, 73, 15, 0, 8, 55, 29, -14, 117, -97, -174, 30, -42, 121, 41, 45, 91, -91, 252, -94, -34, 19, 42, -57, 108, -162, 103, -147, 3, -179, -2, -259, -17, 137, -38, -102, -85, -26, 72, 121, -6, -147, -42, -7, 89, 43, -42, 53}
, {56, -77, -52, 2, 1, -68, 74, 94, -15, -105, -3, -12, 40, 18, -51, 1, -27, -59, -50, 52, 61, -75, -32, 39, 6, 68, 2, -60, 50, -38, -94, 54, -100, 82, 60, -89, -44, 22, -74, 5, 47, 14, 2, -30, -9, -4, -29, 1, -2, 28, -68, -16, 32, 37, 37, 6, 14, -21, -131, -62, -32, 96, -35, -50}
, {29, 19, -126, -72, -45, -12, -14, -26, -34, 38, 28, 97, -26, 11, -98, 84, -25, 145, -33, 63, -34, 4, -57, -10, 16, -136, -136, -47, 3, 47, 47, 83, 62, -53, -37, -56, 115, 5, -118, -26, 119, 52, -93, -15, -51, 34, -66, -180, -121, 83, 11, -26, 46, 106, 9, 69, -119, -71, -44, -11, 83, 62, 67, -43}
}
, {{26, -70, -110, -104, -60, 100, -56, 103, -84, -50, 68, 88, -34, -81, 34, -132, 70, 115, -50, -2, -4, 57, 31, 54, -90, 53, -52, -9, -11, -75, 106, 62, -43, -21, -4, 75, -2, -97, 8, 43, 63, -11, -7, -56, -21, 80, 42, 22, 70, 57, -11, -41, 81, -24, -45, 39, -70, -37, 5, -52, -4, 152, -88, 20}
, {-38, 60, -70, 25, -6, 29, -24, -22, -30, 75, 62, 0, -42, 43, 9, -45, -17, 5, -63, 61, 26, 84, 65, -20, -15, 10, -16, -7, -49, -44, 89, 22, -13, 12, -65, -85, 21, -87, 23, -11, 47, -5, -39, 43, 29, -21, -67, -67, -91, 26, -45, -31, -95, -95, 34, 56, -45, 8, -3, -6, -42, -93, -23, 24}
, {1, -13, 29, 58, -23, 26, -37, 5, -38, 122, -65, 11, -33, -7, 19, -30, -74, -79, 22, 93, -21, -34, 121, 35, -34, -17, 97, 48, -106, -139, 50, 46, -84, 9, 6, -96, -66, 69, 16, -24, -51, -70, 92, -39, 145, -46, -5, 0, -68, 0, -17, -20, -18, -138, -60, -143, -63, -31, 75, 41, -17, -55, -13, 68}
}
}
;

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
/**
  ******************************************************************************
  * @file    batchnorm1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _BATCH_NORMALIZATION_3_H_
#define _BATCH_NORMALIZATION_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       2

typedef int16_t batch_normalization_3_output_type[2][64];

#if 0
void batch_normalization_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  batch_normalization_3_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES

#endif//_BATCH_NORMALIZATION_3_H_
/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "batch_normalization_3.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      64
#define INPUT_SAMPLES       2
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void batch_normalization_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  batch_normalization_3_output_type output) {                // OUT

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
/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const int16_t batch_normalization_3_bias[64] = {-281, -134, -531, -197, -73, -172, -442, -458, -170, -112, -216, -199, -289, -221, -73, -568, -109, -360, -59, -630, 22, -520, -135, -299, -356, -578, -449, -478, -293, -479, -617, -451, -399, -104, -770, -576, -500, -148, -228, -486, -524, -228, -61, -346, -184, -163, -524, -357, 13, -358, -542, -146, -71, -493, -554, -538, -501, -161, -510, -111, -514, -201, 29, -184}
;
const int16_t batch_normalization_3_kernel[64] = {181, 205, 122, 245, 438, 414, 192, 111, 182, 326, 228, 389, 261, 145, 435, 127, 571, 377, 288, 143, 231, 249, 482, 196, 166, 111, 129, 104, 197, 181, 124, 104, 187, 469, 120, 151, 129, 259, 203, 93, 160, 404, 326, 148, 212, 410, 117, 220, 564, 115, 140, 258, 625, 112, 99, 104, 109, 372, 72, 232, 124, 379, 523, 484}
;
/**
  ******************************************************************************
  * @file    maxpool1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _MAX_POOLING1D_3_H_
#define _MAX_POOLING1D_3_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef int16_t max_pooling1d_3_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void max_pooling1d_3(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_MAX_POOLING1D_3_H_
/**
  ******************************************************************************
  * @file    maxpool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "max_pooling1d_3.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS  64
#define INPUT_SAMPLES   2
#define POOL_SIZE       2
#define POOL_STRIDE     2
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_LINEAR

// For fixed point quantization
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void max_pooling1d_3(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  static LONG_NUMBER_T max[INPUT_CHANNELS];

  for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef ACTIVATION_LINEAR
      max[k] = input[pos_x*POOL_STRIDE][k];
      x = 1;
#elif defined(ACTIVATION_RELU)
      max[k] = 0;
      x = 0;
#endif
    }

    for (; x < POOL_SIZE; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        if (max[k] < input[(pos_x * POOL_STRIDE) + x][k])
          max[k] = input[(pos_x * POOL_STRIDE) + x][k];
      }
    }

    for (k = 0; k < INPUT_CHANNELS; k++) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN since using arm_relu_q* is not more efficient, but use SSAT anyway
#if ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR > 0
      output[pos_x][k] = __SSAT(max[k] >> (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#else
      output[pos_x][k] = __SSAT(max[k] << (INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR), sizeof(NUMBER_T) * 8);
#endif
#else
      max[k] = scale(NUMBER_T, max[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, max[k]);
#endif
    }
  }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_LINEAR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    flatten.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define OUTPUT_DIM 64

typedef int16_t flatten_output_type[OUTPUT_DIM];

#if 0
void flatten(
  const number_t input[1][64], 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_FLATTEN_H_
/**
  ******************************************************************************
  * @file    flatten.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0.0
  * @date    26 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "flatten.h"
#include "number.h"
#endif

#define OUTPUT_DIM 64

#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t

static inline void flatten(
  const NUMBER_T input[1][64], 			      // IN
	NUMBER_T output[OUTPUT_DIM]) {			                // OUT

  NUMBER_T *input_flat = (NUMBER_T *)input;

  // Copy data from input to output only if input and output don't point to the same memory address already
  if (input_flat != output) {
    for (size_t i = 0; i < OUTPUT_DIM; i++) {
      output[i] = input_flat[i];
    }
  }
}

#undef OUTPUT_DIM
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_H_
#define _DENSE_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 16

typedef int16_t dense_output_type[FC_UNITS];

#if 0
void dense(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 64
#define FC_UNITS 16
#define ACTIVATION_RELU

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 

    output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);

    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
    output[k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[k] = clamp_to(NUMBER_T, output_acc);
    }
#endif
  }
#else


  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_RELU
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES 64
#define FC_UNITS 16


const int16_t dense_bias[FC_UNITS] = {-185, 17, 6, -156, -65, -76, 6, 7, 13, -147, -22, -140, -153, -104, -66, -67}
;

const int16_t dense_kernel[FC_UNITS][INPUT_SAMPLES] = {{-43, -26, -118, -46, -69, 86, -15, 56, 48, -58, -57, 27, 40, -84, -65, -19, -24, 27, 104, -26, -58, -30, -7, -86, -2, 75, 13, -144, -38, -141, 8, 0, 84, -78, 21, -55, -74, -3, 79, 100, 25, -47, -164, -128, -86, -33, 106, 40, -122, -141, -37, -42, 69, -155, 65, 75, 155, 16, -10, -84, 97, 9, -98, 102}
, {81, -132, 176, 14, -2, -37, 64, 101, -58, 3, -124, 15, -149, 7, -45, 24, 5, 20, -6, -93, -40, -54, -114, -62, -53, -58, -37, -111, -36, 166, -77, 148, -128, -13, -43, -206, -11, -89, -65, -19, -25, -34, 15, 73, 41, 5, 107, -49, -23, -9, -75, -357, -89, -29, -147, 140, 82, -49, -77, -59, 43, 40, -216, -16}
, {1, -79, -9, -88, -28, -127, 143, -81, -185, 71, 105, 10, -216, 95, 17, 17, -71, 39, -32, 48, -139, 141, 33, -233, -95, -37, -92, 145, -15, -53, -36, -81, -181, 96, 55, -159, -125, -232, -30, 85, -101, -20, -49, -118, 11, -2, -140, -49, -50, -129, -31, -245, 41, -56, 8, 235, 82, -177, 111, -247, -101, -9, -256, -83}
, {13, 38, -43, -131, 70, 66, -140, -106, 32, 77, 67, 40, 86, -80, -113, -140, 39, 202, -51, -43, -60, -41, 99, 72, 72, 68, -193, 94, -53, -74, 87, -12, -57, -162, 6, 173, -136, -74, 38, -1, -32, -122, -83, -121, 41, 153, 34, -42, -54, -117, 143, -73, -166, 130, 148, 4, -12, -142, -23, -109, 124, 20, -32, -113}
, {69, -188, 72, -145, -150, 146, 165, -49, -82, 24, -163, 86, -219, 80, 17, 48, -104, 79, -91, 67, -118, 2, 151, -111, 75, 30, -141, 1, 17, 190, 107, -149, -206, 48, -18, 81, -167, -222, -44, 40, -137, -145, -70, -31, 57, -49, -193, 107, -125, 175, 42, 244, -109, 40, -27, 21, 38, -51, -71, 32, 63, 36, -224, -103}
, {-141, -67, 159, 41, -78, 34, 125, 78, -83, 170, -210, 57, -167, -116, -91, 12, -4, -265, -26, -198, 57, 54, -20, -52, -7, -129, 120, -162, -38, 15, 1, 158, -10, -57, 42, -7, 76, -154, -76, -174, 33, -206, -168, 18, -24, 16, 31, -60, -13, 146, -219, -23, -161, 27, -3, 48, 98, 167, -177, 7, -184, -44, 5, -54}
, {76, 74, -87, -115, 35, 5, -140, -143, 125, 2, -13, -27, 22, -434, -85, 169, -35, 3, 15, -37, -52, 22, 8, 105, 99, -234, -109, 118, 79, 2, 142, -5, -46, -62, 197, 14, -89, 3, -313, -115, 30, -212, -62, 48, 67, 81, -51, -48, -6, 17, 91, -23, -57, 127, -181, 50, 26, -157, -170, -76, 49, 70, 1, 7}
, {20, -143, -21, 68, -65, 1, -16, -144, -8, 39, 28, 24, 109, -4, 10, 24, -24, 17, -149, 11, 75, -44, -23, -89, -22, -125, -133, -166, 5, -124, 179, -88, 203, -55, -106, 154, -121, 81, 24, 26, 195, 101, 20, 203, -96, -29, 69, 52, -100, 169, 33, -34, -4, 168, 134, -302, 13, 9, -213, -16, 46, 65, 16, 45}
, {-1, -237, 30, 38, 39, -28, 128, 216, -28, 58, -106, -141, -82, -54, 13, -5, -50, 46, -143, 6, 18, -95, -124, 8, -94, 106, 184, 42, 22, 162, -189, 38, 19, 58, -202, -191, 0, -149, 34, -20, -59, 3, 28, 138, -25, -32, -130, 22, -103, -33, -25, -255, -86, -107, -145, 63, 108, 50, 116, 51, -88, -63, -146, 67}
, {-41, -61, -45, 60, -53, -70, 25, -9, -149, 67, -53, -32, 85, 34, -29, -115, -48, 19, -28, -16, -257, 100, 27, -22, 75, -98, -93, -60, -77, -5, 65, 82, 57, 95, -59, -24, 51, 139, 26, 26, 59, 100, -94, -188, -4, 95, 47, 2, 52, -79, 95, 0, -69, 12, 37, -151, 99, 72, 71, 51, 72, 7, -17, 121}
, {-26, -85, -210, 60, -9, -67, -196, -24, 66, -45, 77, 20, 158, 56, -97, -10, -55, -154, -126, -56, -24, -5, -56, 136, -195, -63, 24, 6, -40, -262, 241, 35, -26, -45, 45, 125, -93, -15, 48, 60, 27, -7, -41, 32, -173, 1, 168, 5, -51, -101, 183, -40, 13, 33, -174, -55, -26, -24, -151, -186, 4, 110, -194, -30}
, {-11, -60, 81, -95, -18, 76, 105, 66, -66, -13, -58, 100, 25, -5, -31, 36, -57, 82, -64, 19, -168, 132, 57, -97, 139, 20, 68, -80, -58, 117, -59, 77, 44, -72, 1, 83, -68, 67, 39, -57, 25, 3, -180, -125, -9, 43, -33, 58, -101, 7, 84, 26, 38, -15, 51, -142, 48, 55, 46, 31, 82, 103, -23, -32}
, {-98, -242, -130, 75, 14, -88, -32, -67, -53, -80, -96, 116, -32, -106, 58, -27, -125, 108, -87, 129, 7, -217, -57, -2, 126, -114, 91, -145, -262, -90, -16, 59, 186, 55, 12, 124, -41, -120, 194, 71, 127, -96, 46, -75, -220, -104, -139, 212, -113, 22, 73, -51, -60, -47, 55, -142, 111, -49, 76, -54, 92, 68, -160, 156}
, {-326, -11, -83, 114, -79, -68, 5, 104, -138, 2, -29, 55, -37, -97, -85, -168, -129, -162, -15, -48, -153, -40, 34, -260, 14, -38, -21, 3, -230, 18, -26, 97, 109, -152, 88, -9, -211, -76, -3, -21, 37, -118, -71, -116, -110, 22, -53, 17, 23, 1, 66, -91, -134, 142, 50, -28, -13, 66, 60, -40, -233, 43, 28, -29}
, {-254, 23, -61, -206, -105, -75, -134, 63, -142, -44, 60, 126, 49, -361, 74, -32, -75, -46, -51, 155, -175, -25, -61, -64, -79, -29, 55, 94, -156, -100, 87, -16, 24, -75, -87, -59, -167, -126, -165, -147, -156, 48, -221, 37, -59, -92, 148, -119, 5, 55, -81, 46, -121, 50, -2, -18, -43, -110, -46, -10, -25, 121, -2, -30}
, {-92, 166, -33, -44, 45, 31, -82, -51, -59, -80, 15, 46, 6, 24, -3, -143, 153, -281, -53, -93, -114, 103, 138, -14, -148, -41, -10, 131, -72, -139, 141, 29, -162, -90, -48, 42, 66, -129, 13, -1, -135, 26, 35, -111, -65, -69, 8, -189, -73, -102, -64, -143, -174, 6, 41, 25, 24, -162, 185, 20, 95, 53, -225, -152}
}
;

#undef INPUT_SAMPLES
#undef FC_UNITS
/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _DENSE_1_H_
#define _DENSE_1_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES 16
#define FC_UNITS 1

typedef int16_t dense_1_output_type[FC_UNITS];

#if 0
void dense_1(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const number_t bias[FC_UNITS],			              // IN

	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_DENSE_1_H_
/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "dense_1.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES 16
#define FC_UNITS 1
#define ACTIVATION_LINEAR

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR 9
#define INPUT_SCALE_FACTOR 9
#define OUTPUT_SCALE_FACTOR 9
#define NUMBER_T int16_t
#define LONG_NUMBER_T int32_t


static inline void dense_1(
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN

	const NUMBER_T bias[FC_UNITS],			              // IN

	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 

    output_acc = scale(NUMBER_T, (LONG_NUMBER_T)bias[k], -INPUT_SCALE_FACTOR);

    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
    output[k] = clamp_to(NUMBER_T, output_acc);
#elif defined(ACTIVATION_RELU)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
      output_acc = scale(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[k] = clamp_to(NUMBER_T, output_acc);
    }
#endif
  }
#else


  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#endif


#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_LINEAR
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
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
