#pragma once

#include "safe_int.h"

#include <cstdint>

namespace emu {

using byte_t = SafeInt<uint8_t>;
using sbyte_t = SafeInt<int8_t>;
using word_t = SafeInt<uint16_t>;
using sword_t = SafeInt<int16_t>;

constexpr inline byte_t get_hi(word_t word) noexcept { return static_cast<byte_t>(word >> 8); }

constexpr inline byte_t get_lo(word_t word) noexcept { return static_cast<byte_t>(word); }

constexpr inline word_t assemble(byte_t lo, byte_t hi)
{
    return static_cast<word_t>(lo) | (static_cast<word_t>(hi) << 8);
}

namespace literals {

constexpr inline byte_t operator""_b(unsigned long long int v) noexcept { return byte_t{v}; }

constexpr inline sbyte_t operator""_sb(unsigned long long int v) noexcept { return sbyte_t{v}; }

constexpr inline word_t operator""_w(unsigned long long int v) noexcept { return word_t{v}; }

constexpr inline sword_t operator""_sw(unsigned long long int v) noexcept { return sword_t{v}; }


} // namespace literals

} // namespace emu
