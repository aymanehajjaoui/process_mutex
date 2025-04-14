/* DataWriter.cpp */

#include "DataWriterCSV.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <type_traits>

template <typename T>
void write_scalar(FILE *file, const T &val)
{
    if constexpr (std::is_same_v<T, float>)
    {
        fprintf(file, "%.6f", val);
    }
    else if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t> || std::is_integral_v<T>)
    {
        fprintf(file, "%d", static_cast<int>(val));
    }
    else
    {
        fprintf(stderr, "Unsupported input type for writing!\n");
        fprintf(file, "ERR");
    }
}

void write_data_csv(Channel &channel, const std::string &filename)
{
    try
    {
        FILE *buffer_output_file = fopen(filename.c_str(), "w");
        if (!buffer_output_file)
        {
            std::cerr << "Error opening buffer output file.\n";
            return;
        }

        while (true)
        {
            std::shared_ptr<data_part_t> part;
            {
                std::unique_lock<std::mutex> lock(channel.mtx);
                channel.cond_write_csv.wait(lock, [&]
                {
                    return !channel.data_queue_csv.empty() || channel.acquisition_done || stop_program.load();
                });

                if (stop_program.load() && channel.acquisition_done && channel.data_queue_csv.empty())
                    break;

                if (!channel.data_queue_csv.empty())
                {
                    part = channel.data_queue_csv.front();
                    channel.data_queue_csv.pop();
                }
                else
                {
                    continue;
                }
            }

            for (size_t k = 0; k < MODEL_INPUT_DIM_0; k++)
            {
                write_scalar(buffer_output_file, part->data[k][0]);
                if (k < MODEL_INPUT_DIM_0 - 1)
                    fprintf(buffer_output_file, ",");
            }

            fprintf(buffer_output_file, "\n");
            fflush(buffer_output_file);

            channel.counters->write_count_csv.fetch_add(1, std::memory_order_relaxed);
        }

        fclose(buffer_output_file);
        std::cout << "Data writing on CSV thread on channel " << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_csv for channel " << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
