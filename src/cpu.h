#pragma once

#include "types.h"

#include <algorithm>
#include <vector>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace emu {

struct CPU {
    word_t PC{};
    byte_t SP{};
    byte_t A{};
    byte_t X{};
    byte_t Y{};
    struct {
        bool N{};
        bool V{};
        bool D{};
        bool I{};
        bool Z{};
        bool C{};

        byte_t get() const noexcept
        {
            using namespace emu::literals;
            return (static_cast<byte_t>(N) << 7)     // Negative
                    | (static_cast<byte_t>(V) << 6)  // Overflow
                    | (1_b << 5)                     // Unused, always 1
                    | (1_b << 4)                     // Break, always 1
                    | (static_cast<byte_t>(D) << 3)  // Decimal mode
                    | (static_cast<byte_t>(I) << 2)  // Interrupt
                    | (static_cast<byte_t>(Z) << 1)  // Zero
                    | (static_cast<byte_t>(C) << 0); // Carry
        }

        void set(byte_t new_SR) noexcept
        {
            using namespace emu::literals;
            N = (new_SR & 0x80_b) != 0_b;
            V = (new_SR & 0x40_b) != 0_b;
            D = (new_SR & 0x08_b) != 0_b;
            I = (new_SR & 0x04_b) != 0_b;
            Z = (new_SR & 0x02_b) != 0_b;
            C = (new_SR & 0x01_b) != 0_b;
        }

    } SR{};
};

class Bus {
public:
    explicit Bus();
    void load_file(std::filesystem::path const& p);
    byte_t read(word_t address) const;
    void write(word_t address, byte_t value);

private:
    std::vector<byte_t> memory_space;
};

} // namespace emu