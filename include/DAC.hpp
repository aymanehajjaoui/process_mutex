/*DAC.hpp*/

#pragma once

#include "Common.hpp"
#include <type_traits>

// Initializes DAC outputs (used before sending data to DAC)
void initialize_DAC();

// Convert a generic model output value to DAC-compatible voltage between -1.0V and +1.0V
template <typename T>
float OutputToVoltage(T value)
{
    if constexpr (std::is_same_v<T, int16_t>)
    {
        return static_cast<float>(value) / 8192.0f; // Scale from int16 [-8192,8191] to [-1.0,1.0]
    }
    else if constexpr (std::is_same_v<T, int8_t>)
    {
        return static_cast<float>(value) / 128.0f; // Scale from int8 [-128,127] to [-1.0,1.0]
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        return value; // Already in correct range
    }
    else
    {
        return static_cast<float>(value); // Fallback for other numeric types
    }
}
