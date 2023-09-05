#pragma once

#include "types.h"

#include <vector>
#include <filesystem>

namespace emu {

// Auxiliar SR structure to ease CPU::SR initialization
struct SR {
    bool N{};
    bool V{};
    bool D{};
    bool I{};
    bool Z{};
    bool C{};
};

struct CPU {
    word_t PC{};
    byte_t SP{};
    byte_t A{};
    byte_t X{};
    byte_t Y{};
    struct SR {
        bool N{};
        bool V{};
        bool D{};
        bool I{};
        bool Z{};
        bool C{};

        constexpr operator byte_t() const noexcept
        {
            using namespace emu::literals;
            return (static_cast<byte_t>(N) << 7) // Negative
                | (static_cast<byte_t>(V) << 6)  // Overflow
                | (1_b << 5)                     // Unused, always 1
                | (1_b << 4)                     // Break, always 1
                | (static_cast<byte_t>(D) << 3)  // Decimal mode
                | (static_cast<byte_t>(I) << 2)  // Interrupt
                | (static_cast<byte_t>(Z) << 1)  // Zero
                | (static_cast<byte_t>(C) << 0); // Carry
        }

        constexpr auto& operator=(byte_t new_SR) noexcept
        {
            using namespace emu::literals;
            N = (new_SR & 0x80_b) != 0_b;
            V = (new_SR & 0x40_b) != 0_b;
            D = (new_SR & 0x08_b) != 0_b;
            I = (new_SR & 0x04_b) != 0_b;
            Z = (new_SR & 0x02_b) != 0_b;
            C = (new_SR & 0x01_b) != 0_b;
            return *this;
        }

        constexpr SR() noexcept = default;
        constexpr SR(byte_t sr) noexcept { *this = sr; }
        constexpr SR(::emu::SR sr) noexcept
            : N{sr.N}
            , V{sr.V}
            , D{sr.D}
            , I{sr.I}
            , Z{sr.Z}
            , C{sr.C}
        {}

        constexpr auto operator<=>(SR const&) const noexcept = default;
    };

    SR SR{};

    constexpr auto operator<=>(CPU const&) const noexcept = default;
};

class Bus {
public:
    explicit Bus();
    void load_file(std::filesystem::path const& p);
    byte_t read(word_t address) const;
    void write(word_t address, byte_t value);

    std::vector<byte_t> memory_space;
};

} // namespace emu
