/*SystemUtils.hpp*/

#pragma once

#include <iostream>
#include <sys/statvfs.h>
#include <chrono>
#include <ctime>
#include <csignal>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>

#include "Common.hpp"

// Disk space check
bool is_disk_space_below_threshold(const char *path, double threshold);

void set_process_affinity(int core_id);

bool set_thread_priority(std::thread &th, int priority);

bool set_thread_affinity(std::thread &th, int core_id);

void signal_handler(int sig);

void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns);

void print_channel_stats(const shared_counters_t *counters);

void folder_manager(const std::string &folder_path);

bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_output_csv, bool &save_output_dac);

void wait_for_barrier(std::atomic<int>& barrier, int total_participants);