/*ADC.cpp*/

#include<ADC.hpp>
#include <iostream>


// Function to initialize data acquisition
void initialize_acq()
{
    rp_AcqReset();
    // Configure Red Pitaya for split trigger mode
    if (rp_AcqSetSplitTrigger(true) != RP_OK)
    {
        std::cerr << "rp_AcqSetSplitTrigger failed!" << std::endl;
    }
    if (rp_AcqSetSplitTriggerPass(true) != RP_OK)
    {
        std::cerr << "rp_AcqSetSplitTriggerPass failed!" << std::endl;
    }

    uint32_t g_adc_axi_start, g_adc_axi_size;
    if (rp_AcqAxiGetMemoryRegion(&g_adc_axi_start, &g_adc_axi_size) != RP_OK)
    {
        std::cerr << "rp_AcqAxiGetMemoryRegion failed!" << std::endl;
        exit(-1);
    }
    std::cout << "Reserved memory Start 0x" << std::hex << g_adc_axi_start << " Size 0x" << std::hex << g_adc_axi_size << std::endl;
    std::cout << std::dec;

    // Set decimation factor for both channels
    if (rp_AcqAxiSetDecimationFactorCh(RP_CH_1, DECIMATION) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetDecimationFactor failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqAxiSetDecimationFactorCh(RP_CH_2, DECIMATION) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetDecimationFactor failed!" << std::endl;
        exit(-1);
    }

    // Get and print sampling rate
    float sampling_rate;
    if (rp_AcqGetSamplingRateHz(&sampling_rate) == RP_OK)
    {
        printf("Current Sampling Rate: %.2f Hz\n", sampling_rate);
    }
    else
    {
        fprintf(stderr, "Failed to get sampling rate\n");
    }

    // Set trigger delay for both channels
    if (rp_AcqAxiSetTriggerDelay(RP_CH_1, 0) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetTriggerDelay channel 1 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqAxiSetTriggerDelay(RP_CH_2, 0) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetTriggerDelay channel 2 failed!" << std::endl;
        exit(-1);
    }

    // Set buffer samples for both channels
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, g_adc_axi_start, DATA_SIZE) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetBuffer RP_CH_1 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqAxiSetBufferSamples(RP_CH_2, g_adc_axi_start + (g_adc_axi_size / 2), DATA_SIZE) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetBuffer RP_CH_2 failed!" << std::endl;
        exit(-1);
    }

    // Enable acquisition on both channels
    if (rp_AcqAxiEnable(RP_CH_1, true) != RP_OK)
    {
        std::cerr << "rp_AcqAxiEnable RP_CH_1 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqAxiEnable(RP_CH_2, true) != RP_OK)
    {
        std::cerr << "rp_AcqAxiEnable RP_CH_2 failed!" << std::endl;
        exit(-1);
    }

    // Set trigger levels for both channels
    if (rp_AcqSetTriggerLevel(RP_T_CH_1, 0) != RP_OK)
    {
        std::cerr << "rp_AcqSetTriggerLevel RP_T_CH_1 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqSetTriggerLevel(RP_T_CH_2, 0) != RP_OK)
    {
        std::cerr << "rp_AcqSetTriggerLevel RP_T_CH_2 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqSetTriggerSrcCh(RP_CH_1, RP_TRIG_SRC_CHA_PE) != RP_OK)
    {
        std::cerr << "rp_AcqSetTriggerSrcCh RP_CH_1 failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqSetTriggerSrcCh(RP_CH_2, RP_TRIG_SRC_CHB_PE) != RP_OK)
    {
        std::cerr << "rp_AcqSetTriggerSrcCh RP_CH_2 failed!" << std::endl;
        exit(-1);
    }

    // Start acquisition on both channels
    if (rp_AcqStartCh(RP_CH_1) != RP_OK)
    {
        std::cerr << "rp_AcqStart failed!" << std::endl;
        exit(-1);
    }
    if (rp_AcqStartCh(RP_CH_2) != RP_OK)
    {
        std::cerr << "rp_AcqStart failed!" << std::endl;
        exit(-1);
    }
}

// Function to clean up resources used by Red Pitaya
void cleanup()
{
    std::cout << "\nReleasing resources\n";
    rp_AcqStopCh(RP_CH_1); // Stop data acquisition for CH1
    rp_AcqStopCh(RP_CH_2); // Stop data acquisition for CH2
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_AcqAxiEnable(RP_CH_2, false);
    rp_Release(); // Release Red Pitaya resources
    std::cout << "Cleanup done." << std::endl;
}
