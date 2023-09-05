#pragma once

#include "cpu.h"

namespace emu {

struct Emulator {
    CPU cpu{};
    Bus bus{};
    uint64_t clock_counter{};

    void run();
};

} // namespace emu
