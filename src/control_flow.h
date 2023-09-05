#pragma once

#include "types.h"
#include "cpu.h"
#include "instruction.h"

#include <map>

namespace emu {

struct control_block {
    std::vector<inst::Instruction> instructions;
    std::optional<word_t> next_not_taken{};
    std::optional<word_t> next_taken{};
    word_t last_addr{};
    bool complete{};
};

[[nodiscard]] std::map<word_t, control_block> build_control_flow(Bus const& bus,
                                                                 word_t entry_point);

} // namespace emu
