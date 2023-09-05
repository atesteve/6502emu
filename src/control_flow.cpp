#define __cpp_concepts 202002L

#include "control_flow.h"
#include "instruction.h"
#include "types.h"

#include <vector>
#include <cassert>
#include <expected>
#include <stdexcept>
#include <unordered_set>
#include <algorithm>

using namespace emu::literals;

namespace emu {

enum class JumpError {
    JUMP_TO_MIDDLE_OF_INSTRUCTION,
};

std::map<word_t, control_block> build_control_flow(Bus const& bus, word_t entry_point)
{
    std::map<word_t, control_block> block_map;
    auto const first_it = block_map.emplace(entry_point, control_block{}).first;
    std::vector incomplete_blocks{first_it};
    std::unordered_set<word_t> complete_blocks;

    auto const prev_safe = [](auto& cont, auto it) {
        if (it == cont.begin()) {
            return cont.end();
        }
        return std::prev(it);
    };

    auto const get_inst_from_address = [&](control_block& block, word_t address) {
        auto ret = std::ranges::upper_bound(
            block.instructions, address, std::less<>{}, &inst::Instruction::get_pc);
        assert(ret != block.instructions.begin());
        return std::prev(ret);
    };

    auto const break_up_block = [&](control_block& block, word_t next_addr) {
        auto const block_instruction_it = get_inst_from_address(block, next_addr);

        auto const new_block_it = block_map.emplace(next_addr, control_block{}).first;
        auto& new_block = new_block_it->second;
        new_block.last_addr = block.last_addr;
        new_block.next_taken = block.next_taken;
        new_block.next_not_taken = block.next_not_taken;
        new_block.complete = true;
        std::copy(block_instruction_it,
                  block.instructions.end(),
                  std::back_inserter(new_block.instructions));

        block.instructions.erase(block_instruction_it, block.instructions.end());
        block.next_taken = std::nullopt;
        block.next_not_taken = next_addr;
        block.last_addr = next_addr - 1_w;
        block.complete = true;

        return &new_block_it->second;
    };

    auto const create_incomplete_block = [&](word_t next_addr) {
        auto const new_block_it = block_map.emplace(next_addr, control_block{}).first;
        incomplete_blocks.push_back(new_block_it);
        return &new_block_it->second;
    };

    auto const get_block = [&](word_t next_addr,
                               word_t src_block_addr) -> std::expected<control_block*, JumpError> {
        auto const it = prev_safe(block_map, block_map.upper_bound(next_addr));

        // Jump before any existing block
        if (it == block_map.cend()) {
            return create_incomplete_block(next_addr);
        }

        auto& block = it->second;

        // Jump to the start of an existing block
        if (it->first == next_addr) {
            return &it->second;
        }

        // Jump into the source block
        if (it->first == src_block_addr) {
            auto const block_instruction_it = get_inst_from_address(block, next_addr);
            if (block_instruction_it->pc != next_addr && next_addr <= block.last_addr) {
                return std::unexpected(JumpError::JUMP_TO_MIDDLE_OF_INSTRUCTION);
            }
            if (next_addr <= block.last_addr + 1_w) {
                return &it->second;
            }
            return create_incomplete_block(next_addr);
        }

        // Jump after an incomplete block
        if (!block.complete) {
            return create_incomplete_block(next_addr);
        }

        auto const block_instruction_it = get_inst_from_address(block, next_addr);

        // Jump into an invalid location of a complete block
        if (block_instruction_it->pc != next_addr && next_addr <= block.last_addr) {
            return std::unexpected(JumpError::JUMP_TO_MIDDLE_OF_INSTRUCTION);
        }

        // Jump after the end of a complete block
        if (next_addr > block.last_addr) {
            return create_incomplete_block(next_addr);
        }

        // Jump into the middle of an existing block, break it up.
        return break_up_block(block, next_addr);
    };

    while (!incomplete_blocks.empty()) {
        auto* block = &incomplete_blocks.back()->second;
        auto const block_entry_addr = incomplete_blocks.back()->first;
        incomplete_blocks.pop_back();

        block->instructions.push_back(inst::Instruction{block_entry_addr, bus});
        while (!block->complete) {
            inst::Instruction const& inst = block->instructions.back();
            block->last_addr = inst.pc + word_t{inst.length} - 1_w;
            auto const not_taken = inst.get_not_taken_addr();
            auto const taken = inst.get_taken_addr();

            if (taken && not_taken) {
                // Branch instruction
                block->next_taken = taken;
                block->next_not_taken = not_taken;

                if (auto taken_dest = get_block(*taken, block->instructions.front().pc);
                    taken_dest.has_value()) {
                    if (*taken != block->instructions.front().pc && *taken_dest == block) {
                        block = break_up_block(*block, *taken);
                    }
                } else {
                    throw std::runtime_error{"Jump into middle of instruction"};
                }

                if (auto not_taken_dest = get_block(*not_taken, block->instructions.front().pc);
                    not_taken_dest.has_value()) {
                    if (*not_taken_dest == block) {
                        create_incomplete_block(*not_taken);
                    }
                } else {
                    throw std::runtime_error{"Jump into middle of instruction"};
                }

                block->complete = true;

            } else if (taken && !not_taken) {
                // Inconditional jump or branch instruction
                block->next_taken = taken;

                if (auto taken_dest = get_block(*taken, block->instructions.front().pc);
                    taken_dest.has_value()) {
                    if (*taken != block->instructions.front().pc && *taken_dest == block) {
                        block = break_up_block(*block, *taken);
                    }
                } else {
                    throw std::runtime_error{"Jump into middle of instruction"};
                }

                block->complete = true;

            } else if (!taken && !not_taken) {
                // Indirect jump or return instruction - destination unknown
                block->complete = true;
            } else {
                // Non jump or branch instruction
                if (auto not_taken_dest = get_block(*not_taken, block->instructions.front().pc);
                    not_taken_dest.has_value()) {
                    if (not_taken_dest != block) {
                        block->next_not_taken = not_taken;
                        block->complete = true;
                    }
                } else {
                    throw std::runtime_error{"Jump into middle of instruction"};
                }
                block->instructions.push_back(inst::Instruction{*not_taken, bus});
            }
        }
    }

    return block_map;
}

} // namespace emu
