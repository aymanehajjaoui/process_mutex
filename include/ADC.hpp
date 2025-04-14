/*ADC.hpp*/

#pragma once

#include "rp.h"          // Red Pitaya API
#include "Common.hpp"    // Shared data structures and global flags

// Initializes the acquisition system (e.g., trigger settings, sampling, etc.)
void initialize_acq();

// Releases resources and performs cleanup related to acquisition
void cleanup();