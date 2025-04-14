/*CNNWriterCSV.hpp*/

#pragma once

#include "Common.hpp"  // Includes Channel struct, model result buffer, etc.

// Logs CNN inference results to a CSV file for the given channel
void log_results_csv(Channel &channel, const std::string &filename);
