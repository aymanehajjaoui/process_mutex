/* SystemUtils.cpp */

#include "SystemUtils.hpp"
#include "Common.hpp"
#include <iostream>
#include <csignal>
#include <thread>
#include <iomanip>
#include <filesystem>
#include <sys/statvfs.h>
#include <sched.h>
#include <unistd.h>

// Check if available disk space is below a given threshold (in bytes)
bool is_disk_space_below_threshold(const char *path, double threshold)
{
    struct statvfs stat;
    if (statvfs(path, &stat) != 0)
    {
        std::cerr << "Error getting filesystem statistics." << std::endl;
        return false;
    }

    double available_space = stat.f_bsize * stat.f_bavail;
    return available_space < threshold;
}

// Set CPU core affinity for the current process
void set_process_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
    {
        std::cerr << "Failed to set process affinity to Core " << core_id << std::endl;
    }
}

// Set real-time priority for a thread
bool set_thread_priority(std::thread &th, int priority)
{
    struct sched_param param;
    param.sched_priority = priority;
    if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &param) != 0)
    {
        std::cerr << "Failed to set thread priority to " << priority << std::endl;
        return false;
    }
    return true;
}

// Pin a thread to a specific CPU core
bool set_thread_affinity(std::thread &th, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset) == 0;
}

// SIGINT handler to initiate graceful shutdown
void signal_handler(int sig)
{
    if (sig == SIGINT)
    {
        std::cout << "^CSIGINT received, initiating graceful shutdown..." << std::endl;
        stop_program.store(true);
        stop_acquisition.store(true);

        // Signal both child processes to exit
        if (pid1 > 0)
            kill(pid1, SIGINT);
        if (pid2 > 0)
            kill(pid2, SIGINT);

        // Wake up all threads (in current process)
        channel1.cond_write_csv.notify_all();
        channel1.cond_model.notify_all();
        channel1.cond_log_csv.notify_all();
        channel1.cond_log_dac.notify_all();

        channel2.cond_write_csv.notify_all();
        channel2.cond_model.notify_all();
        channel2.cond_log_csv.notify_all();
        channel2.cond_log_dac.notify_all();
    }
}

// Print acquisition duration between two timestamps
void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns)
{
    auto duration_ns = end_ns > start_ns ? end_ns - start_ns : 0;
    auto duration_ms = duration_ns / 1'000'000;

    auto minutes = duration_ms / 60000;
    auto seconds = (duration_ms % 60000) / 1000;
    auto ms = duration_ms % 1000;

    std::cout << std::left << std::setw(40) << label + " acquisition time:"
              << minutes << " min " << seconds << " sec " << ms << " ms\n";
}

// Print statistics collected from a channel's counters
void print_channel_stats(const shared_counters_t *counters)
{
    std::cout << "\n====================================\n\n";

    print_duration("Channel 1", counters[0].trigger_time_ns.load(), counters[0].end_time_ns.load());
    print_duration("Channel 2", counters[1].trigger_time_ns.load(), counters[1].end_time_ns.load());

    std::cout << std::left << std::setw(60) << "Total data acquired CH1:" << counters[0].acquire_count.load() << '\n';
    if (save_data_csv)
        std::cout << std::left << std::setw(60) << "Total lines written CH1 to csv:" << counters[0].write_count_csv.load() << '\n';
    if (save_data_dac)
        std::cout << std::left << std::setw(60) << "Total lines written CH1 to DAC_CH1:" << counters[0].write_count_dac.load() << '\n';
    std::cout << std::left << std::setw(60) << "Total model calculated CH1:" << counters[0].model_count.load() << '\n';
    if (save_output_csv)
        std::cout << std::left << std::setw(60) << "Total results logged CH1 to csv file:" << counters[0].log_count_csv.load() << '\n';
    if (save_output_dac)
        std::cout << std::left << std::setw(60) << "Total results written to DAC_CH1:" << counters[0].log_count_dac.load() << '\n';

    std::cout << std::left << std::setw(60) << "Total data acquired CH2:" << counters[1].acquire_count.load() << '\n';
    if (save_data_csv)
        std::cout << std::left << std::setw(60) << "Total lines written CH2 to csv:" << counters[1].write_count_csv.load() << '\n';
    if (save_data_dac)
        std::cout << std::left << std::setw(60) << "Total lines written CH2 to DAC_CH2:" << counters[1].write_count_dac.load() << '\n';
    std::cout << std::left << std::setw(60) << "Total model calculated CH2:" << counters[1].model_count.load() << '\n';
    if (save_output_csv)
        std::cout << std::left << std::setw(60) << "Total results logged CH2 to csv file:" << counters[1].log_count_csv.load() << '\n';
    if (save_output_dac)
        std::cout << std::left << std::setw(60) << "Total results written to DAC_CH2:" << counters[1].log_count_dac.load() << '\n';

    std::cout << "\n====================================\n";
}

// Ensure directory exists or clear its content
void folder_manager(const std::string &folder_path)
{
    namespace fs = std::filesystem;
    fs::path dir_path(folder_path);

    try
    {
        if (fs::exists(dir_path))
        {
            for (const auto &entry : fs::directory_iterator(dir_path))
            {
                try
                {
                    fs::remove_all(entry);
                }
                catch (const fs::filesystem_error &e)
                {
                    std::cerr << "Failed to delete: " << entry.path() << " - " << e.what() << std::endl;
                }
            }
        }
        else
        {
            if (!fs::create_directories(dir_path))
            {
                std::cerr << "Failed to create directory: " << folder_path << std::endl;
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

// Prompt user to select logging/export preferences
bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_output_csv, bool &save_output_dac)
{
    const int max_attempts = 3;

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        int choice;
        std::cout << "Do you want to save acquired data?\n"
                  << " 1. As CSV only\n"
                  << " 2. To DAC only\n"
                  << " 3. Both CSV and DAC\n"
                  << " 4. None\n"
                  << "Enter your choice (1-4): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 4)
        {
            save_data_csv = (choice == 1 || choice == 3);
            save_data_dac = (choice == 2 || choice == 3);
            break;
        }

        std::cerr << "Invalid input. Please enter a number between 1 and 4.\n";
        if (attempt == max_attempts)
            return false;
    }

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        int choice;
        std::cout << "\nChoose what to do with model output:\n"
                  << " 1. Save as CSV only\n"
                  << " 2. Output to DAC only\n"
                  << " 3. Both CSV and DAC\n"
                  << " 4. None\n"
                  << "Enter your choice (1-4): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 4)
        {
            save_output_csv = (choice == 1 || choice == 3);

            // Prevent DAC conflict between raw data and model output
            if (save_data_dac && (choice == 2 || choice == 3))
            {
                save_output_dac = false;
                std::cerr << "\n[Warning] DAC is already used for raw data.\n"
                          << "Model output will NOT be sent to DAC.\n";
            }
            else
            {
                save_output_dac = (choice == 2 || choice == 3);
            }

            return true;
        }

        std::cerr << "Invalid input. Please enter a number between 1 and 4.\n";
        if (attempt == max_attempts)
            return false;
    }

    return true;
}

// Barrier synchronization for processes or threads
void wait_for_barrier(std::atomic<int> &barrier, int total_participants)
{
    barrier.fetch_add(1);
    while (barrier.load() < total_participants)
        std::this_thread::yield();
}
