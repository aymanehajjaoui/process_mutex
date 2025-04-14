/*SystemUtils.hpp*/

#pragma once

#include <iostream>
#include <string>
#include <atomic>
#include <thread>
#include <sys/statvfs.h>  // For disk space checks

#include "Common.hpp"     // For shared_counters_t and definitions

// Check if the available disk space at the given path is below the specified threshold
bool is_disk_space_below_threshold(const char *path, double threshold);

// Set process affinity to a specific core
void set_process_affinity(int core_id);

// Set real-time thread priority
bool set_thread_priority(std::thread &th, int priority);

// Bind thread to a specific CPU core
bool set_thread_affinity(std::thread &th, int core_id);

// Handle SIGINT for graceful shutdown
void signal_handler(int sig);

// Print the duration between two timestamps in nanoseconds
void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns);

// Print acquisition and logging statistics
void print_channel_stats(const shared_counters_t *counters);

// Ensure the existence of a directory, create if missing
void folder_manager(const std::string &folder_path);

// Interactively ask user how to handle input/output configuration
bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_output_csv, bool &save_output_dac);

// Synchronize threads or processes using a shared barrier
void wait_for_barrier(std::atomic<int> &barrier, int total_participants);
