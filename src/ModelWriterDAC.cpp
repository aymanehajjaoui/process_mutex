/* modelWriterDAC.cpp */

#include "ModelWriterDAC.hpp"
#include <iostream>
#include <type_traits>

// DAC output thread for model results
void log_results_dac(Channel &channel, rp_channel_t rp_channel)
{
    try
    {
        while (true)
        {
            model_result_t result;

            {
                // Wait for new results or shutdown signal
                std::unique_lock<std::mutex> lock(channel.mtx);
                channel.cond_log_dac.wait(lock, [&]
                                          { return !channel.result_buffer_dac.empty() || channel.processing_done || stop_program.load(); });

                // Exit if processing is complete and buffer is empty
                if (stop_program.load() && channel.result_buffer_dac.empty())
                    break;

                if (channel.result_buffer_dac.empty())
                    continue;

                result = channel.result_buffer_dac.front();
                channel.result_buffer_dac.pop_front();
            }

            // Convert model output to voltage and send to DAC
            float voltage = OutputToVoltage(result.output[0]);
            voltage = std::clamp(voltage, -1.0f, 1.0f);
            rp_GenAmp(rp_channel, voltage);

            // Update counter
            channel.counters->log_count_dac.fetch_add(1, std::memory_order_relaxed);
        }

        std::cout << "Logging inference results on DAC thread on channel "
                  << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in log_results_dac for channel "
                  << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
