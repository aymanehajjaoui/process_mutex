/*Common.cpp*/

#include "Common.hpp"

Channel channel1;
Channel channel2;

std::atomic<bool> stop_acquisition(false);
std::atomic<bool> stop_program(false);
