/*DAC.cpp*/

#include "DAC.hpp"
#include "rp.h"
#include <cmath>
#include <iostream>
#include <type_traits>

void initialize_DAC()
{
    rp_GenReset();
    rp_GenWaveform(RP_CH_1, RP_WAVEFORM_DC);
    rp_GenWaveform(RP_CH_2, RP_WAVEFORM_DC);
    rp_GenOutEnable(RP_CH_1);
    rp_GenOutEnable(RP_CH_2);
    rp_GenTriggerOnly(RP_CH_1);
    rp_GenTriggerOnly(RP_CH_2);
}

