/*DataAcquisition.hpp*/

#pragma once

#include "ADC.hpp"  // Contains Red Pitaya acquisition initialization and cleanup

// Starts data acquisition loop on the given channel (CH1 or CH2)
void acquire_data(Channel &channel, rp_channel_t rp_channel);
