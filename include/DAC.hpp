/*DAC.hpp*/

#pragma once

#include "Common.hpp"
#include <thread>
#include <atomic>
#include <type_traits>

void initialize_DAC();

// Templated version for flexible output types
template<typename T>
float OutputToVoltage(T value)
{
    if constexpr (std::is_same_v<T, int16_t>) {
        return static_cast<float>(value) / 8192.0f;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return static_cast<float>(value) / 128.0f;
    } else if constexpr (std::is_same_v<T, float>) {
        return value;
    } else {
        return static_cast<float>(value);
    }
}