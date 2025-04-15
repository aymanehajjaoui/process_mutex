### process_mutex
This is a template used to generate code for RedPitaya using a generated model qualia. This version uses 2 processes, one process per channel (CH1 and CH2) that include threads synchronized using mutexes and condition variables for safe access to variables.
### Project structure
```bash
process_mutex/
├── src/
│   ├── SystemUtils.cpp
│   ├── ModelWriterDAC.cpp
│   ├── ModelWriterCSV.cpp
│   ├── ModelProcessing.cpp
│   ├── main.cpp
│   ├── DataWriterDAC.cpp
│   ├── DataWriterCSV.cpp
│   ├── DataAcquisition.cpp
│   ├── DAC.cpp
│   ├── Common.cpp
│   └── ADC.cpp
├── plot.py
├── ModelOutput/
├── Makefile
├── include/
│   ├── SystemUtils.hpp
│   ├── ModelWriterDAC.hpp
│   ├── ModelWriterCSV.hpp
│   ├── ModelProcessing.hpp
│   ├── DataWriterDAC.hpp
│   ├── DataWriterCSV.hpp
│   ├── DataAcquisition.hpp
│   ├── DAC.hpp
│   ├── Common.hpp
│   └── ADC.hpp
├── DataOutput/
└── CMSIS/
    ├── NN/
    │   ├── Source/
    │   │   ├── FullyConnectedFunctions/
    │   │   │   └── arm_fully_connected_q15.c
    │   │   ├── ConvolutionFunctions/
    │   │   │   ├── arm_convolve_HWC_q15_fast_nonsquare.c
    │   │   │   └── arm_convolve_HWC_q15_basic_nonsquare.c
    │   │   └── ActivationFunctions/
    │   │       └── arm_relu_q15.c
    │   └── Include/
    │       ├── arm_nn_types.h
    │       ├── arm_nnsupportfunctions.h
    │       ├── arm_nn_math_types.h
    │       └── arm_nnfunctions.h
    ├── DSP/
    │   └── Include/
    │       └── arm_math_types.h
    └── Core/
        └── Include/
            ├── cmsis_gcc.h
            └── cmsis_compiler.h
```
