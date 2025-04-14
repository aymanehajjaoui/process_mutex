/*modelWriter.cpp*/

#include "ModelWriterCSV.hpp"
#include "DAC.hpp"
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <type_traits>

// Generic output writer for model result
template<typename T>
void write_output(FILE *file, int index, const T &value, double time_ms) {
    if constexpr (std::is_integral<T>::value) {
        fprintf(file, "%d,%d,%.6f\n", index, static_cast<int>(value), time_ms);
    } else if constexpr (std::is_floating_point<T>::value) {
        fprintf(file, "%d,%.6f,%.6f\n", index, value, time_ms);
    } else {
        fprintf(file, "%d,%d,%.6f\n", index, static_cast<int>(value), time_ms); // Fallback
    }
}

void log_results_csv(Channel &channel, const std::string &filename)
{
    try
    {
        FILE *output_file = fopen(filename.c_str(), "w");
        if (!output_file)
        {
            std::cerr << "Error opening output file: " << filename << "\n";
            return;
        }

        int output_index = 1;

        while (true)
        {
            model_result_t result;

            {
                std::unique_lock<std::mutex> lock(channel.mtx);
                channel.cond_log_csv.wait(lock, [&] {
                    return !channel.result_buffer_csv.empty() || channel.processing_done || stop_program.load();
                });

                if (stop_program.load() && channel.processing_done && channel.result_buffer_csv.empty())
                    break;

                if (channel.result_buffer_csv.empty())
                    continue;

                result = channel.result_buffer_csv.front();
                channel.result_buffer_csv.pop_front();
            }

            write_output(output_file, output_index++, result.output[0], result.computation_time);
            fflush(output_file);
            channel.counters->log_count_csv.fetch_add(1, std::memory_order_relaxed);
        }

        fclose(output_file);
        std::cout << "Logging inference results on csv thread on channel " << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in log_results_csv for channel " << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
