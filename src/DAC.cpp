/*DAC.cpp*/

#include "DAC.hpp"
#include <iostream>
#include <cmath>

// Initialize DAC settings for both channels
void initialize_DAC()
{
    // Reset generator settings
    rp_GenReset();

    // Set waveform to DC for both output channels
    rp_GenWaveform(RP_CH_1, RP_WAVEFORM_DC);
    rp_GenWaveform(RP_CH_2, RP_WAVEFORM_DC);

    // Enable output on both channels
    rp_GenOutEnable(RP_CH_1);
    rp_GenOutEnable(RP_CH_2);

    // Configure trigger-only mode for both channels
    rp_GenTriggerOnly(RP_CH_1);
    rp_GenTriggerOnly(RP_CH_2);
}
