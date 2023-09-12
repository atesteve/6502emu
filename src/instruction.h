#pragma once

#include "types.h"
#include "cpu.h"

#include <array>
#include <string>
#include <optional>

namespace emu::inst {

struct Instruction {
    std::array<byte_t, 3> bytes;
    word_t pc;
    uint16_t length;

    std::optional<word_t> get_not_taken_addr() const;
    std::optional<word_t> get_taken_addr() const;
    word_t decode_abs_addr() const;
    bool is_valid() const;
    bool is_jsr() const;
    bool is_call() const;
    bool is_return() const;
    word_t get_pc() const noexcept { return pc; }

    explicit Instruction(word_t pc, Bus const& bus);
    std::size_t run(CPU& cpu, Bus& bus) const;
    std::string disassemble() const;
};

} // namespace emu::inst
