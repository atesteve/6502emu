#include "instruction.h"
#include "cpu.h"
#include "types.h"

#include <cstdint>
#include <optional>
#include <spdlog/spdlog.h>

#include <array>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>

using namespace emu::literals;

namespace emu::inst {

namespace {

constexpr word_t STACK_BASE = 0x100_w;
constexpr word_t VECTOR_BRK_LO = 0xfffe_w;
constexpr word_t VECTOR_BRK_HI = 0xffff_w;

using inst_handler_t = uint64_t (*)(Instruction const&, CPU&, Bus&);

bool same_page(word_t addr_1, word_t addr_2) { return (addr_1 ^ addr_2) <= 0xff_w; }

// Addressing modes

struct addr_mode_result_t {
    word_t addr;
    byte_t op;
    word_t size;
    int cycles;
};

using addr_fn_t = addr_mode_result_t (*)(Instruction const&, CPU const&, Bus const&, bool);

addr_mode_result_t addr_imm(Instruction const& inst, CPU const&, Bus const&, bool)
{
    return {
        .addr = 0_w,
        .op = inst.bytes[1],
        .size = 1_w,
        .cycles = 1,
    };
}

word_t decode_addr_abs(Instruction const& inst) { return assemble(inst.bytes[1], inst.bytes[2]); }

addr_mode_result_t addr_abs(Instruction const& inst, CPU const&, Bus const& bus, bool dereference)
{
    auto const addr = decode_addr_abs(inst);
    auto const op = dereference ? bus.read(addr) : 0_b;

    return {
        .addr = addr,
        .op = op,
        .size = 2_w,
        .cycles = 3,
    };
}

addr_mode_result_t
    addr_abs_X(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const base_addr = decode_addr_abs(inst);
    auto const effective_addr = base_addr + cpu.X;
    auto const op = dereference ? bus.read(effective_addr) : 0_b;

    return {
        .addr = effective_addr,
        .op = op,
        .size = 2_w,
        .cycles = 3 + (!same_page(effective_addr, base_addr) || !dereference),
    };
}

addr_mode_result_t
    addr_abs_Y(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const base_addr = decode_addr_abs(inst);
    auto const effective_addr = base_addr + cpu.Y;
    auto const op = dereference ? bus.read(effective_addr) : 0_b;

    return {
        .addr = effective_addr,
        .op = op,
        .size = 2_w,
        .cycles = 3 + (!same_page(effective_addr, base_addr) || !dereference),
    };
}

addr_mode_result_t
    addr_X_ind(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const zero_page_base = inst.bytes[1];
    auto const zero_page_addr = zero_page_base + cpu.X;
    auto const ind_addr_lo = bus.read(static_cast<word_t>(zero_page_addr));
    auto const ind_addr_hi = bus.read(static_cast<word_t>(zero_page_addr) + 1_w);
    auto const addr = assemble(ind_addr_lo, ind_addr_hi);
    auto const op = dereference ? bus.read(addr) : 0_b;

    return {
        .addr = addr,
        .op = op,
        .size = 1_w,
        .cycles = 5,
    };
}

addr_mode_result_t
    addr_ind_Y(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const zero_page_addr = inst.bytes[1];
    auto const base_addr_lo = bus.read(static_cast<word_t>(zero_page_addr));
    auto const base_addr_hi = bus.read(static_cast<word_t>(zero_page_addr) + 1_w);
    auto const base_addr = assemble(base_addr_lo, base_addr_hi);
    auto const effective_addr = base_addr + cpu.Y;
    auto const op = dereference ? bus.read(effective_addr) : 0_b;

    return {
        .addr = effective_addr,
        .op = op,
        .size = 1_w,
        .cycles = 4 + (!same_page(effective_addr, base_addr) || !dereference),
    };
}

addr_mode_result_t addr_zpg(Instruction const& inst, CPU const&, Bus const& bus, bool dereference)
{
    auto const addr = static_cast<word_t>(inst.bytes[1]);
    auto const op = dereference ? bus.read(addr) : 0_b;
    return {
        .addr = addr,
        .op = op,
        .size = 1_w,
        .cycles = 2,
    };
}

addr_mode_result_t
    addr_zpg_X(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const zero_page_base = inst.bytes[1];
    auto const addr = static_cast<word_t>(zero_page_base + cpu.X);
    auto const op = dereference ? bus.read(addr) : 0_b;
    return {
        .addr = addr,
        .op = op,
        .size = 1_w,
        .cycles = 3,
    };
}

addr_mode_result_t
    addr_zpg_Y(Instruction const& inst, CPU const& cpu, Bus const& bus, bool dereference)
{
    auto const zero_page_base = inst.bytes[1];
    auto const addr = static_cast<word_t>(zero_page_base + cpu.Y);
    auto const op = dereference ? bus.read(addr) : 0_b;
    return {
        .addr = addr,
        .op = op,
        .size = 1_w,
        .cycles = 3,
    };
}

bool is_negative(byte_t value) { return static_cast<int8_t>(value) < 0; }
bool msb(byte_t value) { return is_negative(value); }

void assign_accumulator(CPU& cpu, byte_t value)
{
    cpu.A = value;
    cpu.SR.N = is_negative(value);
    cpu.SR.Z = value == 0_b;
}

void assign_memory(CPU& cpu, Bus& bus, word_t address, byte_t value)
{
    bus.write(address, value);
    cpu.SR.N = is_negative(value);
    cpu.SR.Z = value == 0_b;
}

// Stack operations

void push(CPU& cpu, Bus& bus, byte_t value)
{
    auto const stack_abs_addr = static_cast<word_t>(cpu.SP) + STACK_BASE;
    bus.write(stack_abs_addr, value);
    --cpu.SP;
}

byte_t pop(CPU& cpu, Bus& bus)
{
    ++cpu.SP;
    auto const stack_abs_addr = static_cast<word_t>(cpu.SP) + STACK_BASE;
    return bus.read(stack_abs_addr);
}

// Generic ops

template<typename BinaryOp>
uint64_t bin_logic_op(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    auto const result = BinaryOp{}(cpu.A, op);
    assign_accumulator(cpu, result);

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

void perform_bin_addition(CPU& cpu, byte_t op)
{
    auto const result_s16 = sword_t{sbyte_t{cpu.A}} + sword_t{sbyte_t{op}} + sword_t{cpu.SR.C};
    auto const result_u16 = word_t{cpu.A} + word_t{op} + word_t{cpu.SR.C};
    auto const result = byte_t{result_u16};
    cpu.SR.V = result_s16 > 127_sw || result_s16 < -128_sw;
    cpu.SR.C = result_u16 > 0xff_w;
    assign_accumulator(cpu, result);
}

uint64_t add_op_bin(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    perform_bin_addition(cpu, op);
    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t add_op_dec(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);

    auto const a_lo = cpu.A & 0xf_b;
    auto const a_hi = cpu.A >> 4;

    auto const op_lo = op & 0xf_b;
    auto const op_hi = op >> 4;

    auto const half_dec_add = [](byte_t a, byte_t b, bool c) {
        auto result = a + b + byte_t{c};
        bool carry = false;
        if (result >= 10_b) {
            result -= 10_b;
            carry = true;
        }
        return std::make_pair(result, carry);
    };

    auto const [result_lo, carry_lo] = half_dec_add(a_lo, op_lo, cpu.SR.C);
    auto const [result_hi, carry_hi] = half_dec_add(a_hi, op_hi, carry_lo);

    auto const result = result_lo | (result_hi << 4);

    cpu.SR.V = msb(cpu.A) == msb(op) && msb(result) != msb(op);
    cpu.SR.C = carry_hi;
    assign_accumulator(cpu, result);

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t add_op(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    if (!cpu.SR.D) {
        return add_op_bin(inst, cpu, bus, addr_mode);
    } else {
        return add_op_dec(inst, cpu, bus, addr_mode);
    }
}

uint64_t sub_op_bin(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    perform_bin_addition(cpu, ~op);
    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t sub_op_dec(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);

    auto const a_lo = cpu.A & 0xf_b;
    auto const a_hi = cpu.A >> 4;

    auto const op_lo = op & 0xf_b;
    auto const op_hi = op >> 4;

    auto const half_dec_sub = [](byte_t a, byte_t b, bool c) {
        auto result = a - b - byte_t{!c};
        bool carry = true;
        if (result >= 10_b) {
            result += 10_b;
            carry = false;
        }
        return std::make_pair(result, carry);
    };

    auto const [result_lo, carry_lo] = half_dec_sub(a_lo, op_lo, cpu.SR.C);
    auto const [result_hi, carry_hi] = half_dec_sub(a_hi, op_hi, carry_lo);

    auto const result = result_lo | (result_hi << 4);

    cpu.SR.V = msb(cpu.A) == msb(op) && msb(result) != msb(op);
    cpu.SR.C = carry_hi;
    assign_accumulator(cpu, result);

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t sub_op(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    if (!cpu.SR.D) {
        return sub_op_bin(inst, cpu, bus, addr_mode);
    } else {
        return sub_op_dec(inst, cpu, bus, addr_mode);
    }
}

uint64_t cmp_op(Instruction const& inst, CPU& cpu, Bus& bus, byte_t reg_val, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    auto const result = reg_val - op;
    cpu.SR.C = op <= reg_val;
    cpu.SR.N = is_negative(result);
    cpu.SR.Z = result == 0_b;

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t load_op(Instruction const& inst, CPU& cpu, Bus& bus, byte_t* reg, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    *reg = op;
    cpu.SR.N = is_negative(op);
    cpu.SR.Z = op == 0_b;

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t
    store_op(Instruction const& inst, CPU& cpu, Bus& bus, byte_t reg_value, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, false);
    bus.write(addr, reg_value);

    cpu.PC += 1_w + size;
    return 1 + cycles;
}

uint64_t transfer_op(Instruction const&, CPU& cpu, byte_t src, byte_t* dst, bool update_flags)
{
    *dst = src;
    if (update_flags) {
        cpu.SR.N = is_negative(src);
        cpu.SR.Z = src == 0_b;
    }
    cpu.PC += 1_w;
    return 2;
}

auto shift_left(byte_t b, bool shift_in)
{
    auto const carry = (b & 0x80_b) != 0_b;
    return std::make_pair((b << 1) | byte_t{shift_in}, carry);
};

auto shift_right(byte_t b, bool shift_in)
{
    auto const carry = (b & 1_b) != 0_b;
    return std::make_pair((b >> 1) | (byte_t{shift_in} << 7), carry);
};

using shift_fn_t = std::pair<byte_t, bool> (*)(byte_t b, bool shift_in);

uint64_t shift_mem_op(Instruction const& inst,
                      CPU& cpu,
                      Bus& bus,
                      shift_fn_t shift_fn,
                      bool roll,
                      bool add_extra_cycle,
                      addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    auto const [result, carry] = shift_fn(op, roll ? cpu.SR.C : false);
    cpu.SR.C = carry;
    assign_memory(cpu, bus, addr, result);

    cpu.PC += size + 1_w;
    return cycles + (add_extra_cycle ? 4 : 3);
}

uint64_t shift_A_op(Instruction const&, CPU& cpu, shift_fn_t shift_fn, bool roll)
{
    auto const [result, carry] = shift_fn(cpu.A, roll ? cpu.SR.C : false);
    cpu.SR.C = carry;
    assign_accumulator(cpu, result);

    cpu.PC += 1_w;
    return 2;
}

uint64_t BIT_op(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);

    cpu.SR.N = (op & 0x80_b) != 0_b;
    cpu.SR.V = (op & 0x40_b) != 0_b;
    cpu.SR.Z = (op & cpu.A) == 0_b;

    cpu.PC += size + 1_w;
    return cycles + 1;
}

word_t decode_addr_rel(Instruction const& inst, word_t pc)
{
    auto const offset = static_cast<sbyte_t>(inst.bytes[1]);
    return (pc + offset) + 2_w;
}

template<typename Fn>
uint64_t branch_op(Instruction const& inst, CPU& cpu, Fn&& test_fn)
{
    uint64_t ret = 0;
    if (test_fn()) {
        auto const new_PC = decode_addr_rel(inst, cpu.PC);
        if (same_page(cpu.PC, new_PC)) {
            ret = 3;
        } else {
            ret = 4;
        }
        cpu.PC = new_PC;
    } else {
        cpu.PC += 2_w;
        ret = 2;
    }
    return ret;
}

template<typename Fn>
uint64_t set_flag_op(Instruction const&, CPU& cpu, Fn&& set_fn)
{
    set_fn();
    cpu.PC += 1_w;
    return 2;
}

uint64_t
    inc_dec_mem_op(Instruction const& inst, CPU& cpu, Bus& bus, addr_fn_t addr_mode, bool increase)
{
    auto const [addr, op, size, cycles] = addr_mode(inst, cpu, bus, true);
    auto const result = increase ? op + 1_b : op - 1_b;
    assign_memory(cpu, bus, addr, result);

    cpu.PC += size + 1_w;
    return cycles + 3;
}

uint64_t inc_dec_reg_op(Instruction const&, CPU& cpu, byte_t* reg, bool increase)
{
    if (increase) {
        ++(*reg);
    } else {
        --(*reg);
    }
    cpu.SR.N = is_negative(*reg);
    cpu.SR.Z = *reg == 0_b;

    cpu.PC += 1_w;
    return 2;
}

// Actual instructions

// clang-format off

uint64_t ORA_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_X_ind); }
uint64_t ORA_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_zpg); }
uint64_t ORA_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_imm); }
uint64_t ORA_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_abs); }
uint64_t ORA_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_ind_Y); }
uint64_t ORA_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_zpg_X); }
uint64_t ORA_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_abs_Y); }
uint64_t ORA_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_or<>>(inst, cpu, bus, addr_abs_X); }

uint64_t AND_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_X_ind); }
uint64_t AND_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_zpg); }
uint64_t AND_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_imm); }
uint64_t AND_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_abs); }
uint64_t AND_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_ind_Y); }
uint64_t AND_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_zpg_X); }
uint64_t AND_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_abs_Y); }
uint64_t AND_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_and<>>(inst, cpu, bus, addr_abs_X); }

uint64_t EOR_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_X_ind); }
uint64_t EOR_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_zpg); }
uint64_t EOR_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_imm); }
uint64_t EOR_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_abs); }
uint64_t EOR_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_ind_Y); }
uint64_t EOR_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_zpg_X); }
uint64_t EOR_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_abs_Y); }
uint64_t EOR_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return bin_logic_op<std::bit_xor<>>(inst, cpu, bus, addr_abs_X); }

uint64_t ADC_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return add_op(inst, cpu, bus, addr_X_ind); }
uint64_t ADC_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return add_op(inst, cpu, bus, addr_zpg); }
uint64_t ADC_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return add_op(inst, cpu, bus, addr_abs); }
uint64_t ADC_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return add_op(inst, cpu, bus, addr_imm); }
uint64_t ADC_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return add_op(inst, cpu, bus, addr_ind_Y); }
uint64_t ADC_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return add_op(inst, cpu, bus, addr_zpg_X); }
uint64_t ADC_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return add_op(inst, cpu, bus, addr_abs_Y); }
uint64_t ADC_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return add_op(inst, cpu, bus, addr_abs_X); }

uint64_t SBC_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return sub_op(inst, cpu, bus, addr_X_ind); }
uint64_t SBC_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return sub_op(inst, cpu, bus, addr_zpg); }
uint64_t SBC_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return sub_op(inst, cpu, bus, addr_abs); }
uint64_t SBC_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return sub_op(inst, cpu, bus, addr_imm); }
uint64_t SBC_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return sub_op(inst, cpu, bus, addr_ind_Y); }
uint64_t SBC_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return sub_op(inst, cpu, bus, addr_zpg_X); }
uint64_t SBC_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return sub_op(inst, cpu, bus, addr_abs_Y); }
uint64_t SBC_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return sub_op(inst, cpu, bus, addr_abs_X); }

uint64_t CMP_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.A, addr_X_ind); }
uint64_t CMP_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return cmp_op(inst, cpu, bus, cpu.A, addr_zpg); }
uint64_t CMP_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return cmp_op(inst, cpu, bus, cpu.A, addr_abs); }
uint64_t CMP_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return cmp_op(inst, cpu, bus, cpu.A, addr_imm); }
uint64_t CMP_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.A, addr_ind_Y); }
uint64_t CMP_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.A, addr_zpg_X); }
uint64_t CMP_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.A, addr_abs_Y); }
uint64_t CMP_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.A, addr_abs_X); }

uint64_t CPY_imm(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.Y, addr_imm); }
uint64_t CPY_zpg(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.Y, addr_zpg); }
uint64_t CPY_abs(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.Y, addr_abs); }

uint64_t CPX_imm(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.X, addr_imm); }
uint64_t CPX_zpg(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.X, addr_zpg); }
uint64_t CPX_abs(Instruction const& inst, CPU& cpu, Bus& bus) { return cmp_op(inst, cpu, bus, cpu.X, addr_abs); }

uint64_t ASL_A(Instruction const& inst, CPU& cpu, Bus&)         { return shift_A_op(inst, cpu, shift_left, false); }
uint64_t ASL_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_left, false, false, addr_zpg); }
uint64_t ASL_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_left, false, false, addr_abs); }
uint64_t ASL_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_left, false, false, addr_zpg_X); }
uint64_t ASL_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_left, false, true, addr_abs_X); }

uint64_t LSR_A(Instruction const& inst, CPU& cpu, Bus&)         { return shift_A_op(inst, cpu, shift_right, false); }
uint64_t LSR_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_right, false, false, addr_zpg); }
uint64_t LSR_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_right, false, false, addr_abs); }
uint64_t LSR_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_right, false, false, addr_zpg_X); }
uint64_t LSR_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_right, false, true, addr_abs_X); }

uint64_t ROL_A(Instruction const& inst, CPU& cpu, Bus&)         { return shift_A_op(inst, cpu, shift_left, true); }
uint64_t ROL_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_left, true, false, addr_zpg); }
uint64_t ROL_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_left, true, false, addr_abs); }
uint64_t ROL_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_left, true, false, addr_zpg_X); }
uint64_t ROL_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_left, true, true, addr_abs_X); }

uint64_t ROR_A(Instruction const& inst, CPU& cpu, Bus&)         { return shift_A_op(inst, cpu, shift_right, true); }
uint64_t ROR_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_right, true, false, addr_zpg); }
uint64_t ROR_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return shift_mem_op(inst, cpu, bus, shift_right, true, false, addr_abs); }
uint64_t ROR_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_right, true, false, addr_zpg_X); }
uint64_t ROR_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return shift_mem_op(inst, cpu, bus, shift_right, true, true, addr_abs_X); }

uint64_t BIT_zpg(Instruction const& inst, CPU& cpu, Bus& bus) { return BIT_op(inst, cpu, bus, addr_zpg); }
uint64_t BIT_abs(Instruction const& inst, CPU& cpu, Bus& bus) { return BIT_op(inst, cpu, bus, addr_abs); }

uint64_t BVC_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.V == 0; }); }
uint64_t BVS_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.V == 1; }); }
uint64_t BCC_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.C == 0; }); }
uint64_t BCS_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.C == 1; }); }
uint64_t BPL_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.N == 0; }); }
uint64_t BMI_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.N == 1; }); }
uint64_t BNE_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.Z == 0; }); }
uint64_t BEQ_rel(Instruction const& inst, CPU& cpu, Bus&) { return branch_op(inst, cpu, [&cpu] { return cpu.SR.Z == 1; }); }

uint64_t CLC_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.C = false; }); }
uint64_t CLI_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.I = false; }); }
uint64_t CLD_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.D = false; }); }
uint64_t CLV_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.V = false; }); }

uint64_t SEC_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.C = true; }); }
uint64_t SEI_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.I = true; }); }
uint64_t SED_impl(Instruction const& inst, CPU& cpu, Bus&) { return set_flag_op(inst, cpu, [&cpu] { cpu.SR.D = true; }); }

uint64_t LDA_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.A, addr_X_ind); }
uint64_t LDA_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.A, addr_zpg); }
uint64_t LDA_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.A, addr_imm); }
uint64_t LDA_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.A, addr_abs); }
uint64_t LDA_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.A, addr_ind_Y); }
uint64_t LDA_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.A, addr_zpg_X); }
uint64_t LDA_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.A, addr_abs_Y); }
uint64_t LDA_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.A, addr_abs_X); }

uint64_t LDX_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.X, addr_imm); }
uint64_t LDX_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.X, addr_zpg); }
uint64_t LDX_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.X, addr_abs); }
uint64_t LDX_zpg_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.X, addr_zpg_Y); }
uint64_t LDX_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.X, addr_abs_Y); }

uint64_t LDY_imm(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.Y, addr_imm); }
uint64_t LDY_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.Y, addr_zpg); }
uint64_t LDY_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return load_op(inst, cpu, bus, &cpu.Y, addr_abs); }
uint64_t LDY_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.Y, addr_zpg_X); }
uint64_t LDY_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return load_op(inst, cpu, bus, &cpu.Y, addr_abs_X); }

uint64_t STA_X_ind(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.A, addr_X_ind); }
uint64_t STA_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.A, addr_zpg); }
uint64_t STA_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.A, addr_abs); }
uint64_t STA_ind_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.A, addr_ind_Y); }
uint64_t STA_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.A, addr_zpg_X); }
uint64_t STA_abs_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.A, addr_abs_Y); }
uint64_t STA_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.A, addr_abs_X); }

uint64_t STY_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.Y, addr_zpg); }
uint64_t STY_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.Y, addr_abs); }
uint64_t STY_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.Y, addr_zpg_X); }

uint64_t STX_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.X, addr_zpg); }
uint64_t STX_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return store_op(inst, cpu, bus, cpu.X, addr_abs); }
uint64_t STX_zpg_Y(Instruction const& inst, CPU& cpu, Bus& bus) { return store_op(inst, cpu, bus, cpu.X, addr_zpg_Y); }

uint64_t TXA_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.X, &cpu.A, true); }
uint64_t TAY_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.A, &cpu.Y, true); }
uint64_t TAX_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.A, &cpu.X, true); }
uint64_t TYA_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.Y, &cpu.A, true); }
uint64_t TXS_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.X, &cpu.SP, false); }
uint64_t TSX_impl(Instruction const& inst, CPU& cpu, Bus&) { return transfer_op(inst, cpu, cpu.SP, &cpu.X, true); }

uint64_t INC_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return inc_dec_mem_op(inst, cpu, bus, addr_zpg, true); }
uint64_t INC_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return inc_dec_mem_op(inst, cpu, bus, addr_abs, true); }
uint64_t INC_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return inc_dec_mem_op(inst, cpu, bus, addr_zpg_X, true); }
uint64_t INC_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return inc_dec_mem_op(inst, cpu, bus, addr_abs_X, true); }

uint64_t INY_impl(Instruction const& inst, CPU& cpu, Bus&) { return inc_dec_reg_op(inst, cpu, &cpu.Y, true); }
uint64_t INX_impl(Instruction const& inst, CPU& cpu, Bus&) { return inc_dec_reg_op(inst, cpu, &cpu.X, true); }

uint64_t DEC_zpg(Instruction const& inst, CPU& cpu, Bus& bus)   { return inc_dec_mem_op(inst, cpu, bus, addr_zpg, false); }
uint64_t DEC_abs(Instruction const& inst, CPU& cpu, Bus& bus)   { return inc_dec_mem_op(inst, cpu, bus, addr_abs, false); }
uint64_t DEC_zpg_X(Instruction const& inst, CPU& cpu, Bus& bus) { return inc_dec_mem_op(inst, cpu, bus, addr_zpg_X, false); }
uint64_t DEC_abs_X(Instruction const& inst, CPU& cpu, Bus& bus) { return inc_dec_mem_op(inst, cpu, bus, addr_abs_X, false); }

uint64_t DEY_impl(Instruction const& inst, CPU& cpu, Bus&) { return inc_dec_reg_op(inst, cpu, &cpu.Y, false); }
uint64_t DEX_impl(Instruction const& inst, CPU& cpu, Bus&) { return inc_dec_reg_op(inst, cpu, &cpu.X, false); }

// clang-format on

uint64_t NOP_impl(Instruction const&, CPU& cpu, Bus&)
{
    cpu.PC += 1_w;
    return 2;
}

uint64_t PHP_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    push(cpu, bus, cpu.SR);

    cpu.PC += 1_w;
    return 3;
}

uint64_t PLP_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    auto const new_SR = pop(cpu, bus);
    cpu.SR = new_SR;

    cpu.PC += 1_w;
    return 4;
}

uint64_t PHA_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    push(cpu, bus, cpu.A);
    cpu.PC += 1_w;
    return 3;
}

uint64_t PLA_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    auto const new_a = pop(cpu, bus);
    assign_accumulator(cpu, new_a);
    cpu.PC += 1_w;
    return 4;
}

uint64_t JSR_abs(Instruction const& inst, CPU& cpu, Bus& bus)
{
    auto const addr = assemble(inst.bytes[1], inst.bytes[2]);

    push(cpu, bus, get_hi(cpu.PC + 2_w));
    push(cpu, bus, get_lo(cpu.PC + 2_w));
    cpu.PC = addr;

    return 6;
}

uint64_t JMP_abs(Instruction const& inst, CPU& cpu, Bus&)
{
    cpu.PC = assemble(inst.bytes[1], inst.bytes[2]);
    return 3;
}

uint64_t JMP_ind(Instruction const& inst, CPU& cpu, Bus& bus)
{
    auto const ind_addr = assemble(inst.bytes[1], inst.bytes[2]);
    auto const addr_lo = bus.read(ind_addr);
    auto const addr_hi = bus.read(ind_addr + 1_w);
    auto const addr = assemble(addr_lo, addr_hi);

    cpu.PC = addr;
    return 5;
}

uint64_t RTS_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    auto const new_pc_lo = pop(cpu, bus);
    auto const new_pc_hi = pop(cpu, bus);
    auto const new_pc = assemble(new_pc_lo, new_pc_hi) + 1_w;

    cpu.PC = new_pc;
    return 6;
}

uint64_t BRK_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    auto const new_pc_lo = bus.read(VECTOR_BRK_LO);
    auto const new_pc_hi = bus.read(VECTOR_BRK_HI);
    auto const new_pc = assemble(new_pc_lo, new_pc_hi);

    push(cpu, bus, get_hi(cpu.PC + 2_w));
    push(cpu, bus, get_lo(cpu.PC + 2_w));
    push(cpu, bus, cpu.SR);

    cpu.SR.I = true;
    cpu.PC = new_pc;

    return 7;
}

uint64_t RTI_impl(Instruction const&, CPU& cpu, Bus& bus)
{
    auto const new_sr = pop(cpu, bus);
    auto const new_pc_lo = pop(cpu, bus);
    auto const new_pc_hi = pop(cpu, bus);
    auto const new_pc = assemble(new_pc_lo, new_pc_hi);

    cpu.SR = new_sr;
    cpu.PC = new_pc;

    return 6;
}

uint64_t unimplemented(Instruction const& inst, CPU&, Bus&)
{
    throw std::runtime_error{fmt::format("Unimplemented opcode: {:02x}", inst.bytes[0])};
}

// Disassembly

using diassemble_handler_t = std::string (*)(Instruction const&);

std::string dis_impl(Instruction const&) { return ""; }

std::string dis_abs(Instruction const& inst)
{
    auto const addr = assemble(inst.bytes[1], inst.bytes[2]);
    return fmt::format(" ${:04x}", addr);
}

std::string dis_abs_X(Instruction const& inst)
{
    auto const addr = assemble(inst.bytes[1], inst.bytes[2]);
    return fmt::format(" ${:04x},X", addr);
}

std::string dis_abs_Y(Instruction const& inst)
{
    auto const addr = assemble(inst.bytes[1], inst.bytes[2]);
    return fmt::format(" ${:04x},Y", addr);
}

std::string dis_imm(Instruction const& inst) { return fmt::format(" #${:02x}", inst.bytes[1]); }

std::string dis_ind(Instruction const& inst)
{
    auto const addr = assemble(inst.bytes[1], inst.bytes[2]);
    return fmt::format(" (${:04x})", addr);
}

std::string dis_X_ind(Instruction const& inst)
{
    return fmt::format(" (${:02x},X)", inst.bytes[1]);
}

std::string dis_ind_Y(Instruction const& inst)
{
    return fmt::format(" (${:02x}),Y", inst.bytes[1]);
}

std::string dis_rel(Instruction const& inst)
{
    auto const offset = static_cast<sbyte_t>(inst.bytes[1]);
    auto const dest_addr = (inst.pc + offset) + 2_w;
    return fmt::format(" {:d} (${:04x})", static_cast<sword_t>(offset) + 2_sw, dest_addr);
}


std::string dis_zpg(Instruction const& inst) { return fmt::format(" ${:02x}", inst.bytes[1]); }

std::string dis_zpg_X(Instruction const& inst) { return fmt::format(" ${:02x},X", inst.bytes[1]); }

std::string dis_zpg_Y(Instruction const& inst) { return fmt::format(" ${:02x},Y", inst.bytes[1]); }

std::string dis_unknown(Instruction const&) { return ""; }

// clang-format off

#define _ unimplemented
constinit std::array<inst_handler_t, 256> const instruction_table{
//       -0        -1         -2       -3 -4          -5         -6        -7  -8        -9         -a       -b -c          -d         -e        -f
/* 0- */ BRK_impl, ORA_X_ind, _,        _, _,         ORA_zpg,   ASL_zpg,   _, PHP_impl, ORA_imm,   ASL_A,    _, _,         ORA_abs,   ASL_abs,   _,
/* 1- */ BPL_rel,  ORA_ind_Y, _,        _, _,         ORA_zpg_X, ASL_zpg_X, _, CLC_impl, ORA_abs_Y, _,        _, _,         ORA_abs_X, ASL_abs_X, _,
/* 2- */ JSR_abs,  AND_X_ind, _,        _, BIT_zpg,   AND_zpg,   ROL_zpg,   _, PLP_impl, AND_imm,   ROL_A,    _, BIT_abs,   AND_abs,   ROL_abs,   _,
/* 3- */ BMI_rel,  AND_ind_Y, _,        _, _,         AND_zpg_X, ROL_zpg_X, _, SEC_impl, AND_abs_Y, _,        _, _,         AND_abs_X, ROL_abs_X, _,
/* 4- */ RTI_impl, EOR_X_ind, _,        _, _,         EOR_zpg,   LSR_zpg,   _, PHA_impl, EOR_imm,   LSR_A,    _, JMP_abs,   EOR_abs,   LSR_abs,   _,
/* 5- */ BVC_rel,  EOR_ind_Y, _,        _, _,         EOR_zpg_X, LSR_zpg_X, _, CLI_impl, EOR_abs_Y, _,        _, _,         EOR_abs_X, LSR_abs_X, _,
/* 6- */ RTS_impl, ADC_X_ind, _,        _, _,         ADC_zpg,   ROR_zpg,   _, PLA_impl, ADC_imm,   ROR_A,    _, JMP_ind,   ADC_abs,   ROR_abs,   _,
/* 7- */ BVS_rel,  ADC_ind_Y, _,        _, _,         ADC_zpg_X, ROR_zpg_X, _, SEI_impl, ADC_abs_Y, _,        _, _,         ADC_abs_X, ROR_abs_X, _,
/* 8- */ _,        STA_X_ind, _,        _, STY_zpg,   STA_zpg,   STX_zpg,   _, DEY_impl, _,         TXA_impl, _, STY_abs,   STA_abs,   STX_abs,   _,
/* 9- */ BCC_rel,  STA_ind_Y, _,        _, STY_zpg_X, STA_zpg_X, STX_zpg_Y, _, TYA_impl, STA_abs_Y, TXS_impl, _, _,         STA_abs_X, _,         _,
/* a- */ LDY_imm,  LDA_X_ind, LDX_imm,  _, LDY_zpg,   LDA_zpg,   LDX_zpg,   _, TAY_impl, LDA_imm,   TAX_impl, _, LDY_abs,   LDA_abs,   LDX_abs,   _,
/* b- */ BCS_rel,  LDA_ind_Y, _,        _, LDY_zpg_X, LDA_zpg_X, LDX_zpg_Y, _, CLV_impl, LDA_abs_Y, TSX_impl, _, LDY_abs_X, LDA_abs_X, LDX_abs_Y, _,
/* c- */ CPY_imm,  CMP_X_ind, _,        _, CPY_zpg,   CMP_zpg,   DEC_zpg,   _, INY_impl, CMP_imm,   DEX_impl, _, CPY_abs,   CMP_abs,   DEC_abs,   _,
/* d- */ BNE_rel,  CMP_ind_Y, _,        _, _,         CMP_zpg_X, DEC_zpg_X, _, CLD_impl, CMP_abs_Y, _,        _, _,         CMP_abs_X, DEC_abs_X, _,
/* e- */ CPX_imm,  SBC_X_ind, _,        _, CPX_zpg,   SBC_zpg,   INC_zpg,   _, INX_impl, SBC_imm,   NOP_impl, _, CPX_abs,   SBC_abs,   INC_abs,   _,
/* f- */ BEQ_rel,  SBC_ind_Y, _,        _, _,         SBC_zpg_X, INC_zpg_X, _, SED_impl, SBC_abs_Y, _,        _, _,         SBC_abs_X, INC_abs_X, _,
};
#undef _

#define _ ""
constinit std::array<std::string_view, 256> const mnemonig_table{
//       -0     -1     -2     -3 -4     -5     -6     -7 -8     -9     -a    -b  -c     -d     -e     -f
/* 0- */ "BRK", "ORA", _,     _, _,     "ORA", "ASL", _, "PHP", "ORA", "ASL", _, _,     "ORA", "ASL", _,
/* 1- */ "BPL", "ORA", _,     _, _,     "ORA", "ASL", _, "CLC", "ORA", _,     _, _,     "ORA", "ASL", _,
/* 2- */ "JSR", "AND", _,     _, "BIT", "AND", "ROL", _, "PLP", "AND", "ROL", _, "BIT", "AND", "ROL", _,
/* 3- */ "BMI", "AND", _,     _, _,     "AND", "ROL", _, "SEC", "AND", _,     _, _,     "AND", "ROL", _,
/* 4- */ "RTI", "EOR", _,     _, _,     "EOR", "LSR", _, "PHA", "EOR", "LSR", _, "JMP", "EOR", "LSR", _,
/* 5- */ "BVC", "EOR", _,     _, _,     "EOR", "LSR", _, "CLI", "EOR", _,     _, _,     "EOR", "LSR", _,
/* 6- */ "RTS", "ADC", _,     _, _,     "ADC", "ROR", _, "PLA", "ADC", "ROR", _, "JMP", "ADC", "ROR", _,
/* 7- */ "BVS", "ADC", _,     _, _,     "ADC", "ROR", _, "SEI", "ADC", _,     _, _,     "ADC", "ROR", _,
/* 8- */ _,     "STA", _,     _, "STY", "STA", "STX", _, "DEY", _,     "TXA", _, "STY", "STA", "STX", _,
/* 9- */ "BCC", "STA", _,     _, "STY", "STA", "STX", _, "TYA", "STA", "TXS", _, _,     "STA", _,     _,
/* a- */ "LDY", "LDA", "LDX", _, "LDY", "LDA", "LDX", _, "TAY", "LDA", "TAX", _, "LDY", "LDA", "LDX", _,
/* b- */ "BCS", "LDA", _,     _, "LDY", "LDA", "LDX", _, "CLV", "LDA", "TSX", _, "LDY", "LDA", "LDX", _,
/* c- */ "CPY", "CMP", _,     _, "CPY", "CMP", "DEC", _, "INY", "CMP", "DEX", _, "CPY", "CMP", "DEC", _,
/* d- */ "BNE", "CMP", _,     _, _,     "CMP", "DEC", _, "CLD", "CMP", _,     _, _,     "CMP", "DEC", _,
/* e- */ "CPX", "SBC", _,     _, "CPX", "SBC", "INC", _, "INX", "SBC", "NOP", _, "CPX", "SBC", "INC", _,
/* f- */ "BEQ", "SBC", _,     _, _,     "SBC", "INC", _, "SED", "SBC", _,     _, _,     "SBC", "INC", _,
};
#undef _
#define _ 0
constinit std::array<int, 256> const length_table{
//      -0 -1 -2 -3 -4 -5 -6 -7 -8 -9 -a -b -c -d -e -f
/* 0- */ 1, 2, _, _, _, 2, 2, _, 1, 2, 1, _, _, 3, 3, _,
/* 1- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
/* 2- */ 3, 2, _, _, 2, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* 3- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
/* 4- */ 1, 2, _, _, _, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* 5- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
/* 6- */ 1, 2, _, _, _, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* 7- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
/* 8- */ _, 2, _, _, 2, 2, 2, _, 1, _, 1, _, 3, 3, 3, _,
/* 9- */ 2, 2, _, _, 2, 2, 2, _, 1, 3, 1, _, _, 3, _, _,
/* a- */ 2, 2, 2, _, 2, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* b- */ 2, 2, _, _, 2, 2, 2, _, 1, 3, 1, _, 3, 3, 3, _,
/* c- */ 2, 2, _, _, 2, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* d- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
/* e- */ 2, 2, _, _, 2, 2, 2, _, 1, 2, 1, _, 3, 3, 3, _,
/* f- */ 2, 2, _, _, _, 2, 2, _, 1, 3, _, _, _, 3, 3, _,
};
#undef _
#define _ dis_unknown
constinit std::array<diassemble_handler_t, 256> const disassemble_table{
//       -0        -1         -2       -3 -4          -5         -6        -7  -8        -9         -a       -b -c          -d         -e        -f
/* 0- */ dis_impl, dis_X_ind, _,        _, _,         dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, _,         dis_abs,   dis_abs,   _,
/* 1- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
/* 2- */ dis_abs,  dis_X_ind, _,        _, dis_zpg,   dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* 3- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
/* 4- */ dis_impl, dis_X_ind, _,        _, _,         dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* 5- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
/* 6- */ dis_impl, dis_X_ind, _,        _, _,         dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_ind,   dis_abs,   dis_abs,   _,
/* 7- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
/* 8- */ _,        dis_X_ind, _,        _, dis_zpg,   dis_zpg,   dis_zpg,   _, dis_impl, _,         dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* 9- */ dis_rel,  dis_ind_Y, _,        _, dis_zpg_X, dis_zpg_X, dis_zpg_Y, _, dis_impl, dis_abs_Y, dis_impl, _, _,         dis_abs_X, _,         _,
/* a- */ dis_imm,  dis_X_ind, dis_imm,  _, dis_zpg,   dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* b- */ dis_rel,  dis_ind_Y, _,        _, dis_zpg_X, dis_zpg_X, dis_zpg_Y, _, dis_impl, dis_abs_Y, dis_impl, _, dis_abs_X, dis_abs_X, dis_abs_Y, _,
/* c- */ dis_imm,  dis_X_ind, _,        _, dis_zpg,   dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* d- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
/* e- */ dis_imm,  dis_X_ind, _,        _, dis_zpg,   dis_zpg,   dis_zpg,   _, dis_impl, dis_imm,   dis_impl, _, dis_abs,   dis_abs,   dis_abs,   _,
/* f- */ dis_rel,  dis_ind_Y, _,        _, _,         dis_zpg_X, dis_zpg_X, _, dis_impl, dis_abs_Y, _,        _, _,         dis_abs_X, dis_abs_X, _,
};
#undef _
// clang-format on

} // namespace

Instruction::Instruction(word_t pc_, Bus const& bus)
    : pc{pc_}
{
    bytes[0] = bus.read(pc);
    length = length_table[static_cast<size_t>(bytes[0])];
    for (int i = 1; i < length; ++i) {
        bytes[i] = bus.read(pc + byte_t{i});
    }
}

std::size_t Instruction::run(CPU& cpu, Bus& bus) const
{
    auto const inst_fn = instruction_table[static_cast<size_t>(bytes[0])];
    return inst_fn(*this, cpu, bus);
}

std::string Instruction::disassemble() const
{
    auto const mnemonic = mnemonig_table[static_cast<size_t>(bytes[0])];
    auto const dis_fn = disassemble_table[static_cast<size_t>(bytes[0])];
    std::string ret = fmt::format("{:04x}: ", pc);
    for (int i = 0; i < length; ++i) {
        ret += fmt::format("{:02x} ", bytes[i]);
    }
    for (int i = length; i < 3; ++i) {
        ret += "   ";
    }
    if (!mnemonic.empty()) {
        ret += std::string{mnemonic};
    } else {
        ret += fmt::format("UNK({:02x})", bytes[0]);
    }
    return ret + dis_fn(*this);
}

std::optional<word_t> Instruction::get_not_taken_addr() const
{
    switch (static_cast<uint8_t>(bytes[0])) {
    case 0x4c: // JMP abs
    case 0x6c: // JMP ind
    case 0x40: // RTI
    case 0x60: // RTS
        return std::nullopt;

    default:
        return pc + static_cast<word_t>(length);
    }
}

std::optional<word_t> Instruction::get_taken_addr() const
{
    switch (static_cast<uint8_t>(bytes[0])) {
    case 0x4c: // JMP abs
        return decode_addr_abs(*this);

    case 0x10: // BPL
    case 0x30: // BMI
    case 0x50: // BVC
    case 0x70: // BVS
    case 0x90: // BCC
    case 0xb0: // BCS
    case 0xd0: // BNE
    case 0xf0: // BEQ
        return decode_addr_rel(*this, pc);

    default:
        return std::nullopt;
    }
}

bool Instruction::is_valid() const
{
    return !mnemonig_table[static_cast<size_t>(bytes[0])].empty();
}

bool Instruction::is_jsr() const { return pc == 0x20_w; }

} // namespace emu::inst
