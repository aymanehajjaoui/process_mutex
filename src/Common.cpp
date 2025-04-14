/*Common.cpp*/

#include "Common.hpp"

// Global instances for Channel 1 and Channel 2
Channel channel1;
Channel channel2;

// Atomic flags used for stopping acquisition and program execution
std::atomic<bool> stop_acquisition(false);
std::atomic<bool> stop_program(false);
