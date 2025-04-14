/*Common.hpp*/

#pragma once

// Enable CMSIS-NN and ARM DSP optimizations
#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE

// Standard headers
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

// File system
#include <dirent.h>
#include <sys/stat.h>

// Red Pitaya API
#include "rp.h"

// Model input/output types
#include "../model/include/model.h"

// General constants
#define DATA_SIZE 16384
#define QUEUE_MAX_SIZE 1000000
#define DECIMATION (125000 / MODEL_INPUT_DIM_0)
#define DISK_SPACE_THRESHOLD (0.2 * 1024 * 1024 * 1024) // 200MB
#define SHM_COUNTERS "/channel_counters"

// Thread priorities
#define acq_priority 1
#define write__csv_priority 1
#define write_dac_priority 1
#define model_priority 20
#define log_csv_priority 1
#define log_dac_priority 1

// Global user configuration flags
extern bool save_data_csv;
extern bool save_data_dac;
extern bool save_output_csv;
extern bool save_output_dac;

// Thread stop control flags
extern std::atomic<bool> stop_acquisition;
extern std::atomic<bool> stop_program;

// Structure for a chunk of acquired data
struct data_part_t
{
    input_t data;
};

// Model inference result and timing
struct model_result_t
{
    output_t output;
    double computation_time;
};

// Shared memory structure for tracking runtime statistics
struct shared_counters_t
{
    std::atomic<int> acquire_count;
    std::atomic<int> model_count;
    std::atomic<int> write_count_csv;
    std::atomic<int> write_count_dac;
    std::atomic<int> log_count_csv;
    std::atomic<int> log_count_dac;
    std::atomic<uint64_t> trigger_time_ns;
    std::atomic<uint64_t> end_time_ns;
    std::atomic<int> ready_barrier;
};

// Channel context object (one per channel)
struct Channel
{
    std::queue<std::shared_ptr<data_part_t>> data_queue_csv;
    std::queue<std::shared_ptr<data_part_t>> data_queue_dac;
    std::queue<std::shared_ptr<data_part_t>> model_queue;

    std::deque<model_result_t> result_buffer_csv;
    std::deque<model_result_t> result_buffer_dac;

    std::mutex mtx;
    std::condition_variable cond_write_csv;
    std::condition_variable cond_write_dac;
    std::condition_variable cond_model;
    std::condition_variable cond_log_csv;
    std::condition_variable cond_log_dac;

    rp_acq_trig_state_t state;

    std::chrono::steady_clock::time_point trigger_time_point;
    std::chrono::steady_clock::time_point end_time_point;

    bool acquisition_done = false;
    bool processing_done = false;
    bool channel_triggered = false;

    shared_counters_t *counters = nullptr;

    std::atomic<uint64_t> trigger_time_ns{0};
    std::atomic<uint64_t> end_time_ns{0};

    rp_channel_t channel_id;
};

// Global channels
extern Channel channel1, channel2;

// PIDs for forked processes (used in the process version)
extern pid_t pid1;
extern pid_t pid2;

// Converts raw int16_t Red Pitaya buffer to model input format
template <typename T>
inline void convert_raw_data(const int16_t *src, T dst[MODEL_INPUT_DIM_0][1], size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if constexpr (std::is_same<T, float>::value)
        {
            dst[i][0] = static_cast<float>(src[i]) / 8192.0f; // Normalize float
        }
        else if constexpr (std::is_same<T, int8_t>::value)
        {
            dst[i][0] = static_cast<int8_t>(std::round(src[i] / 64.0f)); // Scale to int8
        }
        else if constexpr (std::is_same<T, int16_t>::value)
        {
            dst[i][0] = src[i]; // Direct copy
        }
        else
        {
            static_assert(!sizeof(T *), "Unsupported data type in convert_raw_data.");
        }
    }
}
