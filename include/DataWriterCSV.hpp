/*DataWriterCSV.hpp*/

#pragma once

#include "Common.hpp" // Provides access to Channel struct and global flags

// Writes acquired raw data to a CSV file from the channel's CSV queue
void write_data_csv(Channel &channel, const std::string &filename);
