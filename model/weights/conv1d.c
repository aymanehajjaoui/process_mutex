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