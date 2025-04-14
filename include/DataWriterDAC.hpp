/*DataWriterDAC.hpp*/

#pragma once

#include "Common.hpp"  // Provides Channel struct and flags
#include "DAC.hpp"     // Provides OutputToVoltage and DAC initialization

// Writes acquired raw data to the DAC from the channel's DAC queue
void write_data_dac(Channel &channel, rp_channel_t rp_channel);
