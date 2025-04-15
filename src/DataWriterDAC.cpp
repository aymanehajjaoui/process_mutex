/* DataWriterDAC.cpp */

#include "DataWriterDAC.hpp"
#include <iostream>

void write_data_dac(Channel &channel, rp_channel_t rp_channel)
{
    try
    {
        while (true)
        {
            std::shared_ptr<data_part_t> part;
            {
                std::unique_lock<std::mutex> lock(channel.mtx);
                channel.cond_write_dac.wait(lock, [&]
                                            { return !channel.data_queue_dac.empty() || channel.acquisition_done || stop_program.load(); });

                if (stop_program.load() && channel.acquisition_done && channel.data_queue_dac.empty())
                    break;

                if (!channel.data_queue_dac.empty())
                {
                    part = channel.data_queue_dac.front();
                    channel.data_queue_dac.pop();
                }
                else
                {
                    continue;
                }
            }

            for (size_t k = 0; k < MODEL_INPUT_DIM_0; k++)
            {
                float voltage = OutputToVoltage(part->data[k][0]);

                voltage = std::clamp(voltage, -1.0f, 1.0f);

                rp_GenAmp(rp_channel, voltage);
            }

            channel.counters->write_count_dac.fetch_add(1, std::memory_order_relaxed);
        }
        std::cout << "Data writing on DAC thread on channel " << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_dac for channel " << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
