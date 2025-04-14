/*CNNWriterDAC.hpp*/

#pragma once

#include "Common.hpp"   // Includes Channel, model result buffer, etc.
#include "DAC.hpp"      // Includes DAC output utilities

// Outputs CNN inference results to DAC for the given channel
void log_results_dac(Channel &channel, rp_channel_t rp_channel);
