/* main.cpp */

#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

#include "rp.h"
#include "Common.hpp"
#include "SystemUtils.hpp"
#include "DataAcquisition.hpp"
#include "DataWriterCSV.hpp"
#include "DataWriterDAC.hpp"
#include "ModelProcessing.hpp"
#include "ModelWriterCSV.hpp"
#include "ModelWriterDAC.hpp"
#include "DAC.hpp"

// Global process IDs and flags for user preferences
pid_t pid1 = -1;
pid_t pid2 = -1;
bool save_data_csv = false;
bool save_data_dac = false;
bool save_output_csv = false;
bool save_output_dac = false;

int main()
{
    if (rp_Init() != RP_OK)
    {
        std::cerr << "Rp API init failed!" << std::endl;
        return -1;
    }

    // Handle Ctrl+C signal for clean shutdown
    std::signal(SIGINT, signal_handler);

    folder_manager("DataOutput");
    folder_manager("ModelOutput");

    // Setup shared memory for counters
    int shm_fd_counters = shm_open(SHM_COUNTERS, O_CREAT | O_RDWR, 0666);
    if (shm_fd_counters == -1 || ftruncate(shm_fd_counters, sizeof(shared_counters_t) * 2) == -1)
    {
        std::cerr << "Error setting up shared memory for counters!" << std::endl;
        return -1;
    }

    auto *shared_counters = static_cast<shared_counters_t *>(mmap(
        nullptr, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_counters, 0));

    if (shared_counters == MAP_FAILED)
    {
        std::cerr << "Shared memory mapping failed!" << std::endl;
        return -1;
    }

    // Initialize atomic counters
    for (int i = 0; i < 2; ++i)
    {
        new (&shared_counters[i].acquire_count) std::atomic<int>(0);
        new (&shared_counters[i].model_count) std::atomic<int>(0);
        new (&shared_counters[i].write_count_csv) std::atomic<int>(0);
        new (&shared_counters[i].write_count_dac) std::atomic<int>(0);
        new (&shared_counters[i].log_count_csv) std::atomic<int>(0);
        new (&shared_counters[i].log_count_dac) std::atomic<int>(0);
        new (&shared_counters[i].ready_barrier) std::atomic<int>(0);
    }

    std::cout << "Starting program" << std::endl;

    // Ask user for input preferences
    if (!ask_user_preferences(save_data_csv, save_data_dac, save_output_csv, save_output_dac))
    {
        std::cerr << "User input failed. Exiting." << std::endl;
        return -1;
    }

    ::save_data_csv = save_data_csv;
    ::save_data_dac = save_data_dac;
    ::save_output_csv = save_output_csv;
    ::save_output_dac = save_output_dac;

    initialize_acq(); // Initialize ADC settings
    initialize_DAC(); // Initialize DAC output

    // ==== Fork Child 1 (Channel 1) ====
    pid1 = fork();
    if (pid1 < 0)
    {
        std::cerr << "Fork for CH1 failed!" << std::endl;
        return -1;
    }
    else if (pid1 == 0)
    {
        std::cout << "Child Process 1 (CH1) started. PID: " << getpid() << std::endl;

        auto *shared_counters_ch1 = static_cast<shared_counters_t *>(mmap(
            nullptr, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_counters, 0));
        if (shared_counters_ch1 == MAP_FAILED)
        {
            std::cerr << "Shared memory mapping failed in CH1!" << std::endl;
            exit(-1);
        }

        channel1.counters = &shared_counters_ch1[0];
        set_process_affinity(0); // Bind to CPU core 0

        wait_for_barrier(shared_counters_ch1[0].ready_barrier, 2);

        std::thread acq_thread(acquire_data, std::ref(channel1), RP_CH_1);
        std::thread model_thread(model_inference, std::ref(channel1));

        std::thread write_thread_csv, write_thread_dac, log_thread_csv, log_thread_dac;
        if (save_data_csv)
            write_thread_csv = std::thread(write_data_csv, std::ref(channel1), "DataOutput/data_ch1.csv");
        if (save_data_dac)
            write_thread_dac = std::thread(write_data_dac, std::ref(channel1), RP_CH_1);
        if (save_output_csv)
            log_thread_csv = std::thread(log_results_csv, std::ref(channel1), "ModelOutput/output_ch1.csv");
        if (save_output_dac)
            log_thread_dac = std::thread(log_results_dac, std::ref(channel1), RP_CH_1);

        set_thread_priority(model_thread, model_priority);

        if (acq_thread.joinable())
            acq_thread.join();
        if (model_thread.joinable())
            model_thread.join();
        if (save_data_csv && write_thread_csv.joinable())
            write_thread_csv.join();
        if (save_data_dac && write_thread_dac.joinable())
            write_thread_dac.join();
        if (save_output_csv && log_thread_csv.joinable())
            log_thread_csv.join();
        if (save_output_dac && log_thread_dac.joinable())
            log_thread_dac.join();

        std::cout << "Child Process 1 (CH1) finished." << std::endl;
        exit(0);
    }

    // ==== Fork Child 2 (Channel 2) ====
    pid2 = fork();
    if (pid2 < 0)
    {
        std::cerr << "Fork for CH2 failed!" << std::endl;
        return -1;
    }
    else if (pid2 == 0)
    {
        std::cout << "Child Process 2 (CH2) started. PID: " << getpid() << std::endl;

        auto *shared_counters_ch2 = static_cast<shared_counters_t *>(mmap(
            nullptr, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_counters, 0));
        if (shared_counters_ch2 == MAP_FAILED)
        {
            std::cerr << "Shared memory mapping failed in CH2!" << std::endl;
            exit(-1);
        }

        channel2.counters = &shared_counters_ch2[1];
        set_process_affinity(1); // Bind to CPU core 1

        wait_for_barrier(shared_counters_ch2[0].ready_barrier, 2);

        std::thread acq_thread(acquire_data, std::ref(channel2), RP_CH_2);
        std::thread model_thread(model_inference, std::ref(channel2));

        std::thread write_thread_csv, write_thread_dac, log_thread_csv, log_thread_dac;
        if (save_data_csv)
            write_thread_csv = std::thread(write_data_csv, std::ref(channel2), "DataOutput/data_ch2.csv");
        if (save_data_dac)
            write_thread_dac = std::thread(write_data_dac, std::ref(channel2), RP_CH_2);
        if (save_output_csv)
            log_thread_csv = std::thread(log_results_csv, std::ref(channel2), "ModelOutput/output_ch2.csv");
        if (save_output_dac)
            log_thread_dac = std::thread(log_results_dac, std::ref(channel2), RP_CH_2);

        set_thread_priority(model_thread, model_priority);

        if (acq_thread.joinable())
            acq_thread.join();
        if (model_thread.joinable())
            model_thread.join();
        if (save_data_csv && write_thread_csv.joinable())
            write_thread_csv.join();
        if (save_data_dac && write_thread_dac.joinable())
            write_thread_dac.join();
        if (save_output_csv && log_thread_csv.joinable())
            log_thread_csv.join();
        if (save_output_dac && log_thread_dac.joinable())
            log_thread_dac.join();

        std::cout << "Child Process 2 (CH2) finished." << std::endl;
        exit(0);
    }

    // Wait for both child processes
    int status;
    waitpid(pid1, &status, 0);
    waitpid(pid2, &status, 0);

    std::cout << "Both child processes finished." << std::endl;

    cleanup(); // Release Red Pitaya
    print_channel_stats(shared_counters);
    shm_unlink(SHM_COUNTERS); // Clean up shared memory

    return 0;
}
