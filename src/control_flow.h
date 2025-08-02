#pragma once

#include "types.h"
#include "cpu.h"
#include "instruction.h"

#include <map>
#include <unordered_set>

namespace emu {

struct control_block {
    // Instructions in the block.
    std::vector<inst::Instruction> instructions;
    // Address of the "not taken" next block.
    std::optional<word_t> next_not_taken{};
    // Address of the "taken" next block.
    std::optional<word_t> next_taken{};
    // Address of the first instruction past the block.
    word_t last_addr{};
    // If true, this block has been completed, i.e., the last instruction is a ret, branch, jump or
    // call instruction.
    bool complete{};
};

[[nodiscard]] std::map<word_t, control_block>
    build_control_flow(Bus const& bus,
                       word_t entry_point,
                       std::unordered_set<word_t>* function_calls = nullptr);

} // namespace emu
