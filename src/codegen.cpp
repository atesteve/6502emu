#include "codegen.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Verifier.h>

#include <llvm/IR/PassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/ModRef.h>

#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <cassert>
#include <memory>
#include <array>
#include <type_traits>
#include <functional>
#include <bit>

using namespace emu::literals;

namespace emu {
namespace {

constexpr word_t VECTOR_BRK_LO = 0xfffe_w;
constexpr word_t VECTOR_BRK_HI = 0xffff_w;

constexpr word_t STACK_BASE = 0x100_w;

enum class RegOffset : uint32_t {
    PC,
    SP,
    A,
    X,
    Y,
    SR,
};
using enum RegOffset;

enum class FlagOffset : uint32_t {
    N,
    V,
    D,
    I,
    Z,
    C,
};
using enum FlagOffset;

struct Context {
    inst::Instruction const* inst{};
    llvm::LLVMContext* context{};
    llvm::IRBuilder<>* builder{};
    llvm::StructType* cpu_type{};
    llvm::Function* fn{};
    llvm::Function* read_bus_fn{};
    llvm::Function* write_bus_fn{};
    llvm::Function* add_fn{};
    llvm::Function* sub_fn{};
    llvm::Function* call_function_fn{};
    llvm::Value* cycle_counter_ptr{};
    llvm::Value* aux_8b_ptr{};
    llvm::Module* module{};
};

using inst_codegen_t = llvm::Value* (*)(Context const&);

// Addressing modes

struct addr_mode_result_t {
    llvm::Value* addr;
    llvm::Value* op;
    llvm::Value* cycles;
};

template<typename Int>
llvm::IntegerType* int_type(llvm::LLVMContext& context)
{
    if constexpr (std::is_same_v<std::decay_t<Int>, bool>) {
        return llvm::IntegerType::get(context, 1);
    } else {
        return llvm::IntegerType::get(context, sizeof(Int) * CHAR_BIT);
    }
}

template<typename Int>
llvm::ConstantInt* int_const(llvm::LLVMContext& context, Int value)
{
    auto* const type = int_type<Int>(context);
    if constexpr (std::is_same_v<std::decay_t<Int>, byte_t>) {
        return llvm::ConstantInt::get(type, static_cast<uint8_t>(value));
    } else if constexpr (std::is_same_v<std::decay_t<Int>, sbyte_t>) {
        return llvm::ConstantInt::get(type, static_cast<int8_t>(value));
    } else if constexpr (std::is_same_v<std::decay_t<Int>, word_t>) {
        return llvm::ConstantInt::get(type, static_cast<uint16_t>(value));
    } else if constexpr (std::is_same_v<std::decay_t<Int>, sword_t>) {
        return llvm::ConstantInt::get(type, static_cast<int16_t>(value));
    } else {
        return llvm::ConstantInt::get(type, value);
    }
}

template<typename Int>
llvm::IntegerType* int_type(Context const& c)
{
    return int_type<Int>(*c.context);
}

template<typename Int>
llvm::ConstantInt* int_const(Context const& c, Int value)
{
    return int_const<Int>(*c.context, value);
}

template<typename Int>
llvm::Value* expect(Context const& c, llvm::Value* value, Int expected_value)
{
    return c.builder->CreateIntrinsic(
        llvm::Intrinsic::expect, {int_type<Int>(c)}, {value, int_const<Int>(c, expected_value)});
}

template<typename Int>
llvm::Value* load(Context const& c, llvm::Value* ptr)
{
    return c.builder->CreateAlignedLoad(int_type<Int>(c), ptr, llvm::Align::Of<Int>());
}

template<typename Int>
void store(Context const& c, llvm::Value* value, llvm::Value* ptr)
{
    c.builder->CreateAlignedStore(value, ptr, llvm::Align::Of<Int>());
}

template<typename Int>
llvm::Value* unaligned_load(Context const& c, llvm::Value* ptr)
{
    return c.builder->CreateAlignedLoad(int_type<Int>(c), ptr, llvm::Align{1});
}

llvm::Value* assemble(Context const& c, llvm::Value* lo, llvm::Value* hi)
{
    auto* const lo_16 = c.builder->CreateZExt(lo, int_type<uint16_t>(c));
    auto* const hi_16 = c.builder->CreateZExt(hi, int_type<uint16_t>(c));
    auto* const hi_shift = c.builder->CreateShl(hi_16, 8);
    return c.builder->CreateOr(lo_16, hi_shift);
}

auto* load_reg(Context const& c, RegOffset offset)
{
    auto* p = c.builder->CreateGEP(
        c.cpu_type, c.fn->getArg(0), {int_const(c, 0), int_const(c, std::to_underlying(offset))});
    return load<uint8_t>(c, p);
}

void store_reg(Context const& c, RegOffset offset, llvm::Value* value)
{
    auto* p = c.builder->CreateGEP(
        c.cpu_type, c.fn->getArg(0), {int_const(c, 0), int_const(c, std::to_underlying(offset))});
    store<uint8_t>(c, value, p);
}

void store_pc(Context const& c, llvm::Value* value)
{
    auto* p = c.builder->CreateGEP(
        c.cpu_type, c.fn->getArg(0), {int_const(c, 0), int_const(c, std::to_underlying(PC))});
    store<uint16_t>(c, value, p);
}

llvm::Value* load_pc(Context const& c)
{
    auto* p = c.builder->CreateGEP(
        c.cpu_type, c.fn->getArg(0), {int_const(c, 0), int_const(c, std::to_underlying(PC))});
    return load<uint16_t>(c, p);
}

auto* load_flag(Context const& c, FlagOffset offset)
{
    auto* p = c.builder->CreateGEP(c.cpu_type,
                                   c.fn->getArg(0),
                                   {int_const(c, 0),
                                    int_const(c, std::to_underlying(SR)),
                                    int_const(c, std::to_underlying(offset))});
    return load<bool>(c, p);
}

void store_flag(Context const& c, FlagOffset offset, llvm::Value* value)
{
    auto* p = c.builder->CreateGEP(c.cpu_type,
                                   c.fn->getArg(0),
                                   {int_const(c, 0),
                                    int_const(c, std::to_underlying(SR)),
                                    int_const(c, std::to_underlying(offset))});
    store<bool>(c, value, p);
}

void return_function(Context const& c, llvm::Value* pc_value)
{
    store_pc(c, pc_value);
    auto* const cycles = load<uint64_t>(c, c.cycle_counter_ptr);
    c.builder->CreateRet(cycles);
}

llvm::Value* read_bus(Context const& c, llvm::Value* addr)
{
    auto* const masked_addr = c.builder->CreateAnd(addr, int_const(c, 0x8000_w));
    auto* const eq_0 =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ, masked_addr, int_const(c, 0_w));

    auto* const regular_memory_block = llvm::BasicBlock::Create(*c.context, "", c.fn);
    auto* const mapped_memory_block = llvm::BasicBlock::Create(*c.context, "", c.fn);
    auto* const continue_block = llvm::BasicBlock::Create(*c.context, "", c.fn);

    c.builder->CreateCondBr(expect(c, eq_0, true), regular_memory_block, mapped_memory_block);

    // If regular memory, just read from memory
    c.builder->SetInsertPoint(regular_memory_block);
    auto* const p = c.builder->CreateGEP(llvm::ArrayType::get(int_type<uint8_t>(c), 0x10000),
                                         c.fn->getArg(2),
                                         {int_const(c, 0), addr});
    auto* const value_read = load<uint8_t>(c, p);
    store<uint8_t>(c, value_read, c.aux_8b_ptr);
    c.builder->CreateBr(continue_block);

    // If mapped memory, perform a call to the bus object
    c.builder->SetInsertPoint(mapped_memory_block);
    auto* const call = c.builder->CreateCall(c.read_bus_fn, {c.fn->getArg(1), addr});
    call->addFnAttr(llvm::Attribute::getWithMemoryEffects(
        *c.context, llvm::MemoryEffects::inaccessibleMemOnly()));
    store<uint8_t>(c, call, c.aux_8b_ptr);
    c.builder->CreateBr(continue_block);

    // Load produced value and continue
    c.builder->SetInsertPoint(continue_block);
    return load<uint8_t>(c, c.aux_8b_ptr);
}

llvm::Value* read_bus(Context const& c, llvm::ConstantInt* addr)
{
    if (addr->getZExtValue() < 0x8000) {
        // If regular memory, just read from memory
        auto* const p = c.builder->CreateGEP(llvm::ArrayType::get(int_type<uint8_t>(c), 0x10000),
                                             c.fn->getArg(2),
                                             {int_const(c, 0), addr});
        return load<uint8_t>(c, p);
    } else {

        // If mapped memory, perform a call to the bus object
        auto* const call = c.builder->CreateCall(c.read_bus_fn, {c.fn->getArg(1), addr});
        call->addFnAttr(llvm::Attribute::getWithMemoryEffects(
            *c.context, llvm::MemoryEffects::inaccessibleMemOnly()));
        return call;
    }
}

void write_bus(Context const& c, llvm::Value* addr, llvm::Value* value)
{
    auto* const masked_addr = c.builder->CreateAnd(addr, int_const(c, 0x8000_w));
    auto* const eq_0 =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ, masked_addr, int_const(c, 0_w));

    auto* const regular_memory_block = llvm::BasicBlock::Create(*c.context, "", c.fn);
    auto* const mapped_memory_block = llvm::BasicBlock::Create(*c.context, "", c.fn);
    auto* const continue_block = llvm::BasicBlock::Create(*c.context, "", c.fn);

    c.builder->CreateCondBr(expect(c, eq_0, true), regular_memory_block, mapped_memory_block);

    // If regular memory, just write to memory
    c.builder->SetInsertPoint(regular_memory_block);
    auto* const p = c.builder->CreateGEP(llvm::ArrayType::get(int_type<uint8_t>(c), 0x10000),
                                         c.fn->getArg(2),
                                         {int_const(c, 0), addr});
    store<uint8_t>(c, value, p);
    c.builder->CreateBr(continue_block);

    // If mapped memory, perform a call to the bus object
    c.builder->SetInsertPoint(mapped_memory_block);
    auto* const call = c.builder->CreateCall(c.write_bus_fn, {c.fn->getArg(1), addr, value});
    call->addFnAttr(llvm::Attribute::getWithMemoryEffects(
        *c.context, llvm::MemoryEffects::inaccessibleMemOnly()));
    c.builder->CreateBr(continue_block);

    // And just continue
    c.builder->SetInsertPoint(continue_block);
}

void add_cycle_counter(Context const& c, llvm::Value* value);

void make_function_call(Context const& c, llvm::Value* addr, word_t next_addr)
{
    store_pc(c, addr);
    c.builder->CreateCall(c.call_function_fn, {c.fn->getArg(3), addr});

    // add_cycle_counter(c, result);

    // After the function call returns, we need to make sure that the PC points to the correct
    // location and so it is safe to continue
    auto* const current_pc = load_pc(c);
    auto* const eq = c.builder->CreateCmp(
        llvm::CmpInst::Predicate::ICMP_EQ, current_pc, int_const(c, next_addr));
    // Hint the optimizer that the most common outcome is that the PC is correct.

    auto* const block_eq =
        llvm::BasicBlock::Create(*c.context, fmt::format("chk_call_{:04x}_eq", next_addr), c.fn);
    auto* const block_ne =
        llvm::BasicBlock::Create(*c.context, fmt::format("chk_call_{:04x}_ne", next_addr), c.fn);

    c.builder->CreateCondBr(expect(c, eq, true), block_eq, block_ne);

    // If the PC after return does not match the expected PC, return
    c.builder->SetInsertPoint(block_ne);
    return_function(c, current_pc);

    // Otherwise, just continue
    c.builder->SetInsertPoint(block_eq);
}

auto* read_1_byte_opcode(Context const& c) { return read_bus(c, int_const(c, c.inst->pc + 1_w)); }

auto* read_2_byte_opcode(Context const& c)
{
    auto* const result_lo = read_bus(c, int_const(c, c.inst->pc + 1_w));
    auto* const result_hi = read_bus(c, int_const(c, c.inst->pc + 2_w));
    return assemble(c, result_lo, result_hi);
}

auto* same_page(Context const& c, llvm::Value* addr_1, llvm::Value* addr_2)
{
    auto* const addr_xor = c.builder->CreateXor(addr_1, addr_2);
    return c.builder->CreateICmp(
        llvm::CmpInst::Predicate::ICMP_ULE, addr_xor, int_const(c, 0xff_w));
}

// Address mode codegen

addr_mode_result_t addr_imm(Context const& c, bool, bool)
{
    return {
        .addr = nullptr,
        .op = read_1_byte_opcode(c),
        .cycles = int_const<uint64_t>(c, 1),
    };
}

addr_mode_result_t addr_abs(Context const& c, bool dereference, bool)
{
    auto* const addr = read_2_byte_opcode(c);
    auto* const op = dereference ? read_bus(c, addr) : nullptr;

    return {
        .addr = addr,
        .op = op,
        .cycles = int_const<uint64_t>(c, 3),
    };
}

addr_mode_result_t addr_abs_X(Context const& c, bool dereference, bool force_extra_cycle)
{
    auto* const base_addr = read_2_byte_opcode(c);
    auto* const x_reg = load_reg(c, X);
    auto* const x_reg_16 = c.builder->CreateZExt(x_reg, int_type<uint16_t>(c));
    auto* const effective_addr = c.builder->CreateAdd(base_addr, x_reg_16);
    llvm::Value* op = nullptr;
    llvm::Value* extra_cycle;

    if (dereference) {
        op = read_bus(c, effective_addr);
    }

    if (force_extra_cycle) {
        extra_cycle = int_const<uint64_t>(c, 1);
    } else {

        auto* const in_same_page = same_page(c, effective_addr, base_addr);
        auto* const not_in_same_page = c.builder->CreateNot(in_same_page);
        extra_cycle = c.builder->CreateZExt(not_in_same_page, int_type<uint64_t>(c));
    }

    return {
        .addr = effective_addr,
        .op = op,
        .cycles = c.builder->CreateAdd(int_const<uint64_t>(c, 3), extra_cycle),
    };
}

addr_mode_result_t addr_abs_Y(Context const& c, bool dereference, bool force_extra_cycle)
{
    auto* const base_addr = read_2_byte_opcode(c);
    auto* const y_reg = load_reg(c, Y);
    auto* const y_reg_16 = c.builder->CreateZExt(y_reg, int_type<uint16_t>(c));
    auto* const effective_addr = c.builder->CreateAdd(base_addr, y_reg_16);

    llvm::Value* op = nullptr;
    llvm::Value* extra_cycle;

    if (dereference) {
        op = read_bus(c, effective_addr);
    }

    if (force_extra_cycle) {
        extra_cycle = int_const<uint64_t>(c, 1);
    } else {
        auto* const in_same_page = same_page(c, effective_addr, base_addr);
        auto* const not_in_same_page = c.builder->CreateNot(in_same_page);
        extra_cycle = c.builder->CreateZExt(not_in_same_page, int_type<uint64_t>(c));
    }

    return {
        .addr = effective_addr,
        .op = op,
        .cycles = c.builder->CreateAdd(int_const<uint64_t>(c, 3), extra_cycle),
    };
}

addr_mode_result_t addr_X_ind(Context const& c, bool dereference, bool)
{
    auto* const zero_page_base = read_1_byte_opcode(c);
    auto* const x_reg = load_reg(c, X);
    auto* const zero_page_addr = c.builder->CreateAdd(zero_page_base, x_reg);
    auto* const zero_page_addr_p1 = c.builder->CreateAdd(zero_page_addr, int_const(c, 1_b));
    auto* const zero_page_addr_16 = c.builder->CreateZExt(zero_page_addr, int_type<uint16_t>(c));
    auto* const zero_page_addr_p1_16 =
        c.builder->CreateZExt(zero_page_addr_p1, int_type<uint16_t>(c));
    auto* const ind_addr_lo = read_bus(c, zero_page_addr_16);
    auto* const ind_addr_hi = read_bus(c, zero_page_addr_p1_16);
    auto* const addr = assemble(c, ind_addr_lo, ind_addr_hi);
    auto* const op = dereference ? read_bus(c, addr) : nullptr;

    return {
        .addr = addr,
        .op = op,
        .cycles = int_const<uint64_t>(c, 5),
    };
}

addr_mode_result_t addr_ind_Y(Context const& c, bool dereference, bool force_extra_cycle)
{
    auto* const zero_page_base = read_1_byte_opcode(c);
    auto* const zero_page_base_16 = c.builder->CreateZExt(zero_page_base, int_type<uint16_t>(c));
    auto* const zero_page_base_16_p1 = c.builder->CreateAdd(zero_page_base_16, int_const(c, 1_w));

    auto* const base_addr_lo = read_bus(c, zero_page_base_16);
    auto* const base_addr_hi = read_bus(c, zero_page_base_16_p1);
    auto* const base_addr = assemble(c, base_addr_lo, base_addr_hi);

    auto* const y_reg = c.builder->CreateZExt(load_reg(c, Y), int_type<uint16_t>(c));
    auto* const effective_addr = c.builder->CreateAdd(base_addr, y_reg);

    llvm::Value* op = nullptr;
    llvm::Value* extra_cycle;

    if (dereference) {
        op = read_bus(c, effective_addr);
    }

    if (force_extra_cycle) {
        extra_cycle = int_const<uint64_t>(c, 1);
    } else {
        auto* const in_same_page = same_page(c, effective_addr, base_addr);
        auto* const not_in_same_page = c.builder->CreateNot(in_same_page);
        extra_cycle = c.builder->CreateZExt(not_in_same_page, int_type<uint64_t>(c));
    }

    return {
        .addr = effective_addr,
        .op = op,
        .cycles = c.builder->CreateAdd(int_const<uint64_t>(c, 4), extra_cycle),
    };
}

addr_mode_result_t addr_zpg(Context const& c, bool dereference, bool)
{
    auto* const addr = c.builder->CreateZExt(read_1_byte_opcode(c), int_type<uint16_t>(c));
    auto* const op = dereference ? read_bus(c, addr) : nullptr;
    return {
        .addr = addr,
        .op = op,
        .cycles = int_const<uint64_t>(c, 2),
    };
}

addr_mode_result_t addr_zpg_X(Context const& c, bool dereference, bool)
{
    auto* const zero_page_base = read_1_byte_opcode(c);
    auto* const x_reg = load_reg(c, X);
    auto* const addr = c.builder->CreateAdd(zero_page_base, x_reg);
    auto* const addr_16 = c.builder->CreateZExt(addr, int_type<uint16_t>(c));
    auto* const op = dereference ? read_bus(c, addr_16) : nullptr;
    return {
        .addr = addr_16,
        .op = op,
        .cycles = int_const<uint64_t>(c, 3),
    };
}

addr_mode_result_t addr_zpg_Y(Context const& c, bool dereference, bool)
{
    auto* const zero_page_base = read_1_byte_opcode(c);
    auto* const y_reg = load_reg(c, Y);
    auto* const addr = c.builder->CreateAdd(zero_page_base, y_reg);
    auto* const addr_16 = c.builder->CreateZExt(addr, int_type<uint16_t>(c));
    auto* const op = dereference ? read_bus(c, addr_16) : nullptr;
    return {
        .addr = addr_16,
        .op = op,
        .cycles = int_const<uint64_t>(c, 3),
    };
}

// Utility

llvm::Value* is_negative(Context const& c, llvm::Value* value)
{
    return c.builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_SLT, value, int_const(c, 0_b));
}

void set_n_z(Context const& c, llvm::Value* value)
{
    store_flag(c, N, is_negative(c, value));
    auto* const eq_zero =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ, value, int_const(c, 0_b));
    store_flag(c, Z, eq_zero);
}

void add_cycle_counter(Context const& c, llvm::Value* value)
{
    auto* const counter = load<uint64_t>(c, c.cycle_counter_ptr);
    auto* const new_counter = c.builder->CreateAdd(counter, value);
    store<uint64_t>(c, new_counter, c.cycle_counter_ptr);
}

using addr_fn_t = addr_mode_result_t (*)(Context const& c,
                                         bool dereference,
                                         bool force_add_extra_cycle);
using build_bin_fn_t = llvm::Value* (llvm::IRBuilder<>::*)(llvm::Value*,
                                                           llvm::Value*,
                                                           const llvm::Twine&);

void push(Context const& c, llvm::Value* value)
{
    auto* const sp = load_reg(c, SP);
    auto* const sp_16 = c.builder->CreateZExt(sp, int_type<uint16_t>(c));
    auto* const stack_addr = c.builder->CreateAdd(sp_16, int_const(c, STACK_BASE));
    write_bus(c, stack_addr, value);
    auto* const decreased_sp = c.builder->CreateSub(sp, int_const(c, 1_b));
    store_reg(c, SP, decreased_sp);
}

llvm::Value* pop(Context const& c)
{
    auto* const sp = load_reg(c, SP);
    auto* const increased_sp = c.builder->CreateAdd(sp, int_const(c, 1_b));
    store_reg(c, SP, increased_sp);
    auto* const increased_sp_16 = c.builder->CreateZExt(increased_sp, int_type<uint16_t>(c));
    auto* const stack_addr = c.builder->CreateAdd(increased_sp_16, int_const(c, STACK_BASE));
    return read_bus(c, stack_addr);
}

llvm::Value* load_sr(Context const& c)
{
    auto* const n_flag = c.builder->CreateZExt(load_flag(c, N), int_type<uint8_t>(c));
    auto* const v_flag = c.builder->CreateZExt(load_flag(c, V), int_type<uint8_t>(c));
    auto* const d_flag = c.builder->CreateZExt(load_flag(c, D), int_type<uint8_t>(c));
    auto* const i_flag = c.builder->CreateZExt(load_flag(c, I), int_type<uint8_t>(c));
    auto* const z_flag = c.builder->CreateZExt(load_flag(c, Z), int_type<uint8_t>(c));
    auto* const c_shift = c.builder->CreateZExt(load_flag(c, C), int_type<uint8_t>(c));

    auto* const n_shift = c.builder->CreateShl(n_flag, 7);
    auto* const v_shift = c.builder->CreateShl(v_flag, 6);
    auto* const d_shift = c.builder->CreateShl(d_flag, 3);
    auto* const i_shift = c.builder->CreateShl(i_flag, 2);
    auto* const z_shift = c.builder->CreateShl(z_flag, 1);

    auto* const sr_0 = int_const(c, 0x30_b);
    auto* const sr_1 = c.builder->CreateOr(sr_0, n_shift);
    auto* const sr_2 = c.builder->CreateOr(sr_1, v_shift);
    auto* const sr_3 = c.builder->CreateOr(sr_2, d_shift);
    auto* const sr_4 = c.builder->CreateOr(sr_3, i_shift);
    auto* const sr_5 = c.builder->CreateOr(sr_4, z_shift);
    auto* const sr_6 = c.builder->CreateOr(sr_5, c_shift);

    return sr_6;
}

void store_sr(Context const& c, llvm::Value* value)
{
    auto* const n_flag = c.builder->CreateAnd(value, 1 << 7);
    auto* const v_flag = c.builder->CreateAnd(value, 1 << 6);
    auto* const d_flag = c.builder->CreateAnd(value, 1 << 3);
    auto* const i_flag = c.builder->CreateAnd(value, 1 << 2);
    auto* const z_flag = c.builder->CreateAnd(value, 1 << 1);
    auto* const c_flag = c.builder->CreateAnd(value, 1 << 0);

    auto* const n_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, n_flag, int_const(c, 0_b));
    auto* const v_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, v_flag, int_const(c, 0_b));
    auto* const d_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, d_flag, int_const(c, 0_b));
    auto* const i_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, i_flag, int_const(c, 0_b));
    auto* const z_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, z_flag, int_const(c, 0_b));
    auto* const c_shift = c.builder->CreateICmp(llvm::CmpInst::ICMP_NE, c_flag, int_const(c, 0_b));

    store_flag(c, N, n_shift);
    store_flag(c, V, v_shift);
    store_flag(c, D, d_shift);
    store_flag(c, I, i_shift);
    store_flag(c, Z, z_shift);
    store_flag(c, C, c_shift);
}

// Generic instruction code gen

llvm::Value* bin_logic_op(Context const& c, build_bin_fn_t bin_fn, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);
    auto* const a_reg = load_reg(c, A);
    auto* const result = std::invoke(bin_fn, c.builder, a_reg, op, "");
    store_reg(c, A, result);
    set_n_z(c, result);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

auto shift_left(Context const& c, llvm::Value* val, llvm::Value* shift_in)
{
    auto* const result_tmp = c.builder->CreateShl(val, 1);
    auto* const carry_bit = c.builder->CreateAnd(val, int_const(c, 0x80_b));
    auto* const carry_flag =
        c.builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_NE, carry_bit, int_const(c, 0_b));
    auto* const shift_in_u8 = c.builder->CreateZExt(shift_in, int_type<uint8_t>(c));
    auto* const result = c.builder->CreateOr(result_tmp, shift_in_u8);
    return std::make_pair(result, carry_flag);
}

auto shift_right(Context const& c, llvm::Value* val, llvm::Value* shift_in)
{
    auto* const result_tmp = c.builder->CreateLShr(val, 1);
    auto* const carry_bit = c.builder->CreateAnd(val, int_const(c, 1_b));
    auto* const carry_flag = c.builder->CreateTrunc(carry_bit, int_type<bool>(c));
    auto* const shift_in_u8 = c.builder->CreateZExt(shift_in, int_type<uint8_t>(c));
    auto* const shift_in_left = c.builder->CreateShl(shift_in_u8, 7);
    auto* const result = c.builder->CreateOr(result_tmp, shift_in_left);
    return std::make_pair(result, carry_flag);
}

using shift_fn_t = std::pair<llvm::Value*, llvm::Value*> (*)(Context const& c,
                                                             llvm::Value* value,
                                                             llvm::Value* shift_in);

llvm::Value* shift_mem_op(Context const& c, shift_fn_t fn, bool roll, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, true, true);
    auto* const shift_in = roll ? static_cast<llvm::Value*>(load_flag(c, C))
                                : static_cast<llvm::Value*>(int_const(c, false));
    auto const [result, carry_flag] = fn(c, op, shift_in);
    store_flag(c, C, carry_flag);
    write_bus(c, addr, result);
    set_n_z(c, result);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 3)));
    return nullptr;
}

llvm::Value* shift_A_op(Context const& c, shift_fn_t fn, bool roll)
{
    auto* const a_reg = load_reg(c, A);
    auto* const shift_in = roll ? static_cast<llvm::Value*>(load_flag(c, C))
                                : static_cast<llvm::Value*>(int_const(c, false));
    auto const [result, carry_flag] = fn(c, a_reg, shift_in);
    store_flag(c, C, carry_flag);
    store_reg(c, A, result);
    set_n_z(c, result);
    add_cycle_counter(c, int_const<uint64_t>(c, 2));
    return nullptr;
}

llvm::Value* branch_op(Context const& c, FlagOffset flag_off, bool test)
{
    auto const taken_addr = *c.inst->get_taken_addr();
    auto const current_pc = c.inst->pc;
    bool const same_page = (taken_addr & 0xff00_w) == (current_pc & 0xff00_w);

    auto* const flag = load_flag(c, flag_off);
    auto* const test_result =
        c.builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_EQ, flag, int_const(c, test));
    auto* const cycles_base = int_const<uint64_t>(c, 2);

    llvm::Value* cycles;

    if (same_page) {
        // Add 1 cycle if the branch is taken
        cycles = c.builder->CreateAdd(cycles_base,
                                      c.builder->CreateZExt(test_result, int_type<uint64_t>(c)));
    } else {
        // Add 2 cycles if the branch is taken
        auto* const add_cycles =
            c.builder->CreateShl(c.builder->CreateZExt(test_result, int_type<uint64_t>(c)), 1);
        cycles = c.builder->CreateAdd(cycles_base, add_cycles);
    }

    // store_pc(c, int_const(c, c.inst->pc));
    add_cycle_counter(c, cycles);

    return test_result;
}

llvm::Value* cmp_op(Context const& c, RegOffset reg_offset, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);
    auto* const reg = load_reg(c, reg_offset);
    auto* const result = c.builder->CreateSub(reg, op);

    store_flag(c, C, c.builder->CreateICmp(llvm::CmpInst::Predicate::ICMP_ULE, op, reg));
    set_n_z(c, result);

    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

llvm::Value* transfer_op(Context const& c, RegOffset src_offset, RegOffset dst_offset)
{
    auto* const src_reg = load_reg(c, src_offset);
    store_reg(c, dst_offset, src_reg);

    if (dst_offset != SP) {
        set_n_z(c, src_reg);
    }

    add_cycle_counter(c, int_const<uint64_t>(c, 2));
    return nullptr;
}

llvm::Value* set_flag_op(Context const& c, FlagOffset flag_offset, bool value)
{
    store_flag(c, flag_offset, int_const(c, value));
    add_cycle_counter(c, int_const<uint64_t>(c, 2));
    return nullptr;
}

llvm::Value* inc_dec_mem_op(Context const& c, addr_fn_t addr_mode, bool increase)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);
    auto* const result = increase ? c.builder->CreateAdd(op, int_const(c, 1_b))
                                  : c.builder->CreateSub(op, int_const(c, 1_b));
    write_bus(c, addr, result);
    set_n_z(c, result);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 3)));
    return nullptr;
}

llvm::Value* inc_dec_reg_op(Context const& c, RegOffset reg_offset, bool increase)
{
    auto* const reg = load_reg(c, reg_offset);
    auto* const result = increase ? c.builder->CreateAdd(reg, int_const(c, 1_b))
                                  : c.builder->CreateSub(reg, int_const(c, 1_b));
    store_reg(c, reg_offset, result);
    set_n_z(c, result);
    add_cycle_counter(c, int_const<uint64_t>(c, 2));
    return nullptr;
}

llvm::Value* store_op(Context const& c, RegOffset reg_offset, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, false, true);
    auto* const reg = load_reg(c, reg_offset);
    write_bus(c, addr, reg);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

llvm::Value* load_op(Context const& c, RegOffset reg_offset, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);
    store_reg(c, reg_offset, op);
    set_n_z(c, op);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

llvm::Value* add_sub_op(Context const& c, addr_fn_t addr_mode, bool add)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);
    auto* const fn = add ? c.add_fn : c.sub_fn;
    auto* const call = c.builder->CreateCall(fn, {c.fn->getArg(0), op});
    call->setCallingConv(llvm::CallingConv::Fast);
    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

llvm::Value* BIT_op(Context const& c, addr_fn_t addr_mode)
{
    auto const [addr, op, cycles] = addr_mode(c, true, false);

    auto* const n_bit = c.builder->CreateAnd(op, int_const(c, 0x80_b));
    auto* const new_n =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_NE, n_bit, int_const(c, 0_b));

    auto* const v_bit = c.builder->CreateAnd(op, int_const(c, 0x40_b));
    auto* const new_v =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_NE, v_bit, int_const(c, 0_b));

    auto* const a_reg = load_reg(c, A);
    auto* const a_masked = c.builder->CreateAnd(a_reg, op);
    auto* const new_z =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ, a_masked, int_const(c, 0_b));

    store_flag(c, N, new_n);
    store_flag(c, V, new_v);
    store_flag(c, Z, new_z);

    add_cycle_counter(c, c.builder->CreateAdd(cycles, int_const<uint64_t>(c, 1)));
    return nullptr;
}

// Actual instruction codegen functions

llvm::Value* unimplemented(Context const& c)
{
    spdlog::info("Unknown instruction: {}", c.inst->disassemble());
    return nullptr;
}

// clang-format off
llvm::Value* ORA_X_ind(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_X_ind); }
llvm::Value* ORA_zpg(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_zpg); }
llvm::Value* ORA_imm(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_imm); }
llvm::Value* ORA_abs(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_abs); }
llvm::Value* ORA_ind_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_ind_Y); }
llvm::Value* ORA_zpg_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_zpg_X); }
llvm::Value* ORA_abs_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_abs_Y); }
llvm::Value* ORA_abs_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateOr, addr_abs_X); }

llvm::Value* AND_X_ind(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_X_ind); }
llvm::Value* AND_zpg(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_zpg); }
llvm::Value* AND_imm(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_imm); }
llvm::Value* AND_abs(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_abs); }
llvm::Value* AND_ind_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_ind_Y); }
llvm::Value* AND_zpg_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_zpg_X); }
llvm::Value* AND_abs_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_abs_Y); }
llvm::Value* AND_abs_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateAnd, addr_abs_X); }

llvm::Value* EOR_X_ind(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_X_ind); }
llvm::Value* EOR_zpg(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_zpg); }
llvm::Value* EOR_imm(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_imm); }
llvm::Value* EOR_abs(Context const& c)   { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_abs); }
llvm::Value* EOR_ind_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_ind_Y); }
llvm::Value* EOR_zpg_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_zpg_X); }
llvm::Value* EOR_abs_Y(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_abs_Y); }
llvm::Value* EOR_abs_X(Context const& c) { return bin_logic_op(c, &llvm::IRBuilder<>::CreateXor, addr_abs_X); }

llvm::Value* ASL_A(Context const& c)     { return shift_A_op(c, shift_left, false); }
llvm::Value* ASL_zpg(Context const& c)   { return shift_mem_op(c, shift_left, false, addr_zpg); }
llvm::Value* ASL_abs(Context const& c)   { return shift_mem_op(c, shift_left, false, addr_abs); }
llvm::Value* ASL_zpg_X(Context const& c) { return shift_mem_op(c, shift_left, false, addr_zpg_X); }
llvm::Value* ASL_abs_X(Context const& c) { return shift_mem_op(c, shift_left, false, addr_abs_X); }

llvm::Value* LSR_A(Context const& c)     { return shift_A_op(c, shift_right, false); }
llvm::Value* LSR_zpg(Context const& c)   { return shift_mem_op(c, shift_right, false, addr_zpg); }
llvm::Value* LSR_abs(Context const& c)   { return shift_mem_op(c, shift_right, false, addr_abs); }
llvm::Value* LSR_zpg_X(Context const& c) { return shift_mem_op(c, shift_right, false, addr_zpg_X); }
llvm::Value* LSR_abs_X(Context const& c) { return shift_mem_op(c, shift_right, false, addr_abs_X); }

llvm::Value* ROL_A(Context const& c)     { return shift_A_op(c, shift_left, true); }
llvm::Value* ROL_zpg(Context const& c)   { return shift_mem_op(c, shift_left, true, addr_zpg); }
llvm::Value* ROL_abs(Context const& c)   { return shift_mem_op(c, shift_left, true, addr_abs); }
llvm::Value* ROL_zpg_X(Context const& c) { return shift_mem_op(c, shift_left, true, addr_zpg_X); }
llvm::Value* ROL_abs_X(Context const& c) { return shift_mem_op(c, shift_left, true, addr_abs_X); }

llvm::Value* ROR_A(Context const& c)     { return shift_A_op(c, shift_right, true); }
llvm::Value* ROR_zpg(Context const& c)   { return shift_mem_op(c, shift_right, true, addr_zpg); }
llvm::Value* ROR_abs(Context const& c)   { return shift_mem_op(c, shift_right, true, addr_abs); }
llvm::Value* ROR_zpg_X(Context const& c) { return shift_mem_op(c, shift_right, true, addr_zpg_X); }
llvm::Value* ROR_abs_X(Context const& c) { return shift_mem_op(c, shift_right, true, addr_abs_X); }

llvm::Value* ADC_X_ind(Context const& c) { return add_sub_op(c, addr_X_ind, true); }
llvm::Value* ADC_zpg(Context const& c)   { return add_sub_op(c, addr_zpg,   true); }
llvm::Value* ADC_imm(Context const& c)   { return add_sub_op(c, addr_imm,   true); }
llvm::Value* ADC_abs(Context const& c)   { return add_sub_op(c, addr_abs,   true); }
llvm::Value* ADC_ind_Y(Context const& c) { return add_sub_op(c, addr_ind_Y, true); }
llvm::Value* ADC_zpg_X(Context const& c) { return add_sub_op(c, addr_zpg_X, true); }
llvm::Value* ADC_abs_Y(Context const& c) { return add_sub_op(c, addr_abs_Y, true); }
llvm::Value* ADC_abs_X(Context const& c) { return add_sub_op(c, addr_abs_X, true); }

llvm::Value* SBC_X_ind(Context const& c) { return add_sub_op(c, addr_X_ind, false); }
llvm::Value* SBC_zpg(Context const& c)   { return add_sub_op(c, addr_zpg,   false); }
llvm::Value* SBC_imm(Context const& c)   { return add_sub_op(c, addr_imm,   false); }
llvm::Value* SBC_abs(Context const& c)   { return add_sub_op(c, addr_abs,   false); }
llvm::Value* SBC_ind_Y(Context const& c) { return add_sub_op(c, addr_ind_Y, false); }
llvm::Value* SBC_zpg_X(Context const& c) { return add_sub_op(c, addr_zpg_X, false); }
llvm::Value* SBC_abs_Y(Context const& c) { return add_sub_op(c, addr_abs_Y, false); }
llvm::Value* SBC_abs_X(Context const& c) { return add_sub_op(c, addr_abs_X, false); }

llvm::Value* CMP_X_ind(Context const& c) { return cmp_op(c, A, addr_X_ind); }
llvm::Value* CMP_zpg(Context const& c)   { return cmp_op(c, A, addr_zpg); }
llvm::Value* CMP_imm(Context const& c)   { return cmp_op(c, A, addr_imm); }
llvm::Value* CMP_abs(Context const& c)   { return cmp_op(c, A, addr_abs); }
llvm::Value* CMP_ind_Y(Context const& c) { return cmp_op(c, A, addr_ind_Y); }
llvm::Value* CMP_zpg_X(Context const& c) { return cmp_op(c, A, addr_zpg_X); }
llvm::Value* CMP_abs_Y(Context const& c) { return cmp_op(c, A, addr_abs_Y); }
llvm::Value* CMP_abs_X(Context const& c) { return cmp_op(c, A, addr_abs_X); }

llvm::Value* CPX_imm(Context const& c) { return cmp_op(c, X, addr_imm); }
llvm::Value* CPX_zpg(Context const& c) { return cmp_op(c, X, addr_zpg); }
llvm::Value* CPX_abs(Context const& c) { return cmp_op(c, X, addr_abs); }

llvm::Value* CPY_imm(Context const& c) { return cmp_op(c, Y, addr_imm); }
llvm::Value* CPY_zpg(Context const& c) { return cmp_op(c, Y, addr_zpg); }
llvm::Value* CPY_abs(Context const& c) { return cmp_op(c, Y, addr_abs); }

llvm::Value* STA_X_ind(Context const& c) { return store_op(c, A, addr_X_ind); }
llvm::Value* STA_zpg(Context const& c)   { return store_op(c, A, addr_zpg); }
llvm::Value* STA_abs(Context const& c)   { return store_op(c, A, addr_abs); }
llvm::Value* STA_ind_Y(Context const& c) { return store_op(c, A, addr_ind_Y); }
llvm::Value* STA_zpg_X(Context const& c) { return store_op(c, A, addr_zpg_X); }
llvm::Value* STA_abs_Y(Context const& c) { return store_op(c, A, addr_abs_Y); }
llvm::Value* STA_abs_X(Context const& c) { return store_op(c, A, addr_abs_X); }

llvm::Value* STX_zpg(Context const& c)   { return store_op(c, X, addr_zpg); }
llvm::Value* STX_abs(Context const& c)   { return store_op(c, X, addr_abs); }
llvm::Value* STX_zpg_Y(Context const& c) { return store_op(c, X, addr_zpg_Y); }

llvm::Value* STY_zpg(Context const& c)   { return store_op(c, Y, addr_zpg); }
llvm::Value* STY_abs(Context const& c)   { return store_op(c, Y, addr_abs); }
llvm::Value* STY_zpg_X(Context const& c) { return store_op(c, Y, addr_zpg_X); }

llvm::Value* LDA_X_ind(Context const& c) { return load_op(c, A, addr_X_ind); }
llvm::Value* LDA_zpg(Context const& c)   { return load_op(c, A, addr_zpg); }
llvm::Value* LDA_imm(Context const& c)   { return load_op(c, A, addr_imm); }
llvm::Value* LDA_abs(Context const& c)   { return load_op(c, A, addr_abs); }
llvm::Value* LDA_ind_Y(Context const& c) { return load_op(c, A, addr_ind_Y); }
llvm::Value* LDA_zpg_X(Context const& c) { return load_op(c, A, addr_zpg_X); }
llvm::Value* LDA_abs_Y(Context const& c) { return load_op(c, A, addr_abs_Y); }
llvm::Value* LDA_abs_X(Context const& c) { return load_op(c, A, addr_abs_X); }

llvm::Value* LDX_imm(Context const& c)   { return load_op(c, X, addr_imm); }
llvm::Value* LDX_zpg(Context const& c)   { return load_op(c, X, addr_zpg); }
llvm::Value* LDX_abs(Context const& c)   { return load_op(c, X, addr_abs); }
llvm::Value* LDX_zpg_Y(Context const& c) { return load_op(c, X, addr_zpg_Y); }
llvm::Value* LDX_abs_Y(Context const& c) { return load_op(c, X, addr_abs_Y); }

llvm::Value* LDY_imm(Context const& c)   { return load_op(c, Y, addr_imm); }
llvm::Value* LDY_zpg(Context const& c)   { return load_op(c, Y, addr_zpg); }
llvm::Value* LDY_abs(Context const& c)   { return load_op(c, Y, addr_abs); }
llvm::Value* LDY_zpg_X(Context const& c) { return load_op(c, Y, addr_zpg_X); }
llvm::Value* LDY_abs_X(Context const& c) { return load_op(c, Y, addr_abs_X); }

llvm::Value* INC_zpg(Context const& c)   { return inc_dec_mem_op(c, addr_zpg,   true); }
llvm::Value* INC_abs(Context const& c)   { return inc_dec_mem_op(c, addr_abs,   true); }
llvm::Value* INC_zpg_X(Context const& c) { return inc_dec_mem_op(c, addr_zpg_X, true); }
llvm::Value* INC_abs_X(Context const& c) { return inc_dec_mem_op(c, addr_abs_X, true); }

llvm::Value* INX_impl(Context const& c) { return inc_dec_reg_op(c, X, true); }
llvm::Value* INY_impl(Context const& c) { return inc_dec_reg_op(c, Y, true); }

llvm::Value* DEC_zpg(Context const& c)   { return inc_dec_mem_op(c, addr_zpg,   false); }
llvm::Value* DEC_abs(Context const& c)   { return inc_dec_mem_op(c, addr_abs,   false); }
llvm::Value* DEC_zpg_X(Context const& c) { return inc_dec_mem_op(c, addr_zpg_X, false); }
llvm::Value* DEC_abs_X(Context const& c) { return inc_dec_mem_op(c, addr_abs_X, false); }

llvm::Value* DEX_impl(Context const& c) { return inc_dec_reg_op(c, X, false); }
llvm::Value* DEY_impl(Context const& c) { return inc_dec_reg_op(c, Y, false); }

llvm::Value* BIT_zpg(Context const& c) { return BIT_op(c, addr_zpg); }
llvm::Value* BIT_abs(Context const& c) { return BIT_op(c, addr_abs); }

llvm::Value* BVC_rel(Context const& c) { return branch_op(c, V, false); }
llvm::Value* BVS_rel(Context const& c) { return branch_op(c, V, true);  }
llvm::Value* BPL_rel(Context const& c) { return branch_op(c, N, false); }
llvm::Value* BMI_rel(Context const& c) { return branch_op(c, N, true);  }
llvm::Value* BCC_rel(Context const& c) { return branch_op(c, C, false); }
llvm::Value* BCS_rel(Context const& c) { return branch_op(c, C, true);  }
llvm::Value* BNE_rel(Context const& c) { return branch_op(c, Z, false); }
llvm::Value* BEQ_rel(Context const& c) { return branch_op(c, Z, true);  }

llvm::Value* TAX_impl(Context const& c) { return transfer_op(c, A, X); }
llvm::Value* TXA_impl(Context const& c) { return transfer_op(c, X, A); }
llvm::Value* TAY_impl(Context const& c) { return transfer_op(c, A, Y); }
llvm::Value* TYA_impl(Context const& c) { return transfer_op(c, Y, A); }
llvm::Value* TXS_impl(Context const& c) { return transfer_op(c, X, SP); }
llvm::Value* TSX_impl(Context const& c) { return transfer_op(c, SP, X); }

llvm::Value* SEC_impl(Context const& c) { return set_flag_op(c, C, true); }
llvm::Value* SEI_impl(Context const& c) { return set_flag_op(c, I, true); }
llvm::Value* SED_impl(Context const& c) { return set_flag_op(c, D, true); }

llvm::Value* CLC_impl(Context const& c) { return set_flag_op(c, C, false); }
llvm::Value* CLI_impl(Context const& c) { return set_flag_op(c, I, false); }
llvm::Value* CLV_impl(Context const& c) { return set_flag_op(c, V, false); }
llvm::Value* CLD_impl(Context const& c) { return set_flag_op(c, D, false); }

// clang-format on

llvm::Value* PHA_impl(Context const& c)
{
    auto* const a_reg = load_reg(c, A);
    push(c, a_reg);
    add_cycle_counter(c, int_const<uint64_t>(c, 3));
    return nullptr;
}

llvm::Value* PLA_impl(Context const& c)
{
    auto* const new_a = pop(c);
    store_reg(c, A, new_a);
    set_n_z(c, new_a);
    add_cycle_counter(c, int_const<uint64_t>(c, 4));
    return nullptr;
}

llvm::Value* PHP_impl(Context const& c)
{
    auto* const sr_reg = load_sr(c);
    push(c, sr_reg);
    add_cycle_counter(c, int_const<uint64_t>(c, 3));
    return nullptr;
}

llvm::Value* PLP_impl(Context const& c)
{
    auto* const new_sr = pop(c);
    store_sr(c, new_sr);
    add_cycle_counter(c, int_const<uint64_t>(c, 4));
    return nullptr;
}

llvm::Value* RTS_impl(Context const& c)
{
    auto* const new_pc_lo = c.builder->CreateZExt(pop(c), int_type<uint16_t>(c));
    auto* const new_pc_hi = c.builder->CreateZExt(pop(c), int_type<uint16_t>(c));
    auto* const new_pc_m1 = assemble(c, new_pc_lo, new_pc_hi);
    auto* const new_pc = c.builder->CreateAdd(new_pc_m1, int_const(c, 1_w));
    add_cycle_counter(c, int_const<uint64_t>(c, 6));
    return new_pc;
}

llvm::Value* RTI_impl(Context const& c)
{
    auto* const new_sr = pop(c);
    auto* const new_pc_lo = c.builder->CreateZExt(pop(c), int_type<uint16_t>(c));
    auto* const new_pc_hi = c.builder->CreateZExt(pop(c), int_type<uint16_t>(c));
    auto* const new_pc = assemble(c, new_pc_lo, new_pc_hi);
    store_sr(c, new_sr);
    add_cycle_counter(c, int_const<uint64_t>(c, 6));
    return new_pc;
}

llvm::Value* JMP_abs(Context const& c)
{
    add_cycle_counter(c, int_const<uint64_t>(c, 3));
    return int_const(c, *c.inst->get_taken_addr());
}

llvm::Value* JMP_ind(Context const& c)
{
    auto* const addr = read_2_byte_opcode(c);
    auto* const addr_p1 = c.builder->CreateAdd(addr, int_const(c, 1_w));
    auto* const new_pc_lo = read_bus(c, addr);
    auto* const new_pc_hi = read_bus(c, addr_p1);
    auto* const new_pc = assemble(c, new_pc_lo, new_pc_hi);
    add_cycle_counter(c, int_const<uint64_t>(c, 5));
    return new_pc;
}

llvm::Value* JSR_abs(Context const& c)
{
    auto* const new_pc = read_2_byte_opcode(c);

    auto const next_pc = c.inst->pc + 2_w;
    push(c, int_const(c, static_cast<byte_t>(next_pc >> 8)));
    push(c, int_const(c, static_cast<byte_t>(next_pc)));

    add_cycle_counter(c, int_const<uint64_t>(c, 6));

    return new_pc;
}

llvm::Value* BRK_impl(Context const& c)
{
    auto* const new_pc_lo = read_bus(c, int_const(c, VECTOR_BRK_LO));
    auto* const new_pc_hi = read_bus(c, int_const(c, VECTOR_BRK_HI));
    auto* const new_pc = assemble(c, new_pc_lo, new_pc_hi);

    auto const next_pc = c.inst->pc + 2_w;
    push(c, int_const(c, static_cast<byte_t>(next_pc >> 8)));
    push(c, int_const(c, static_cast<byte_t>(next_pc)));
    push(c, load_sr(c));

    store_flag(c, I, int_const(c, true));

    add_cycle_counter(c, int_const<uint64_t>(c, 7));

    return new_pc;
}

llvm::Value* NOP_impl(Context const& c)
{
    add_cycle_counter(c, int_const<uint64_t>(c, 2));
    return nullptr;
}

// clang-format off
#define _ unimplemented
constinit std::array<inst_codegen_t, 256> const instruction_table{
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
#define _ 0
// The table below only works in little-endian systems
static_assert(std::endian::native == std::endian::little,
              "Big or mixed endian architectures not supported yet");
constinit std::array<uint16_t, 256> const self_mod_table{
//       -0      -1    -2    -3 -4    -5    -6    -7 -8    -9    -a    -b -c    -d    -e    -f
/* 0- */ 0xff,   0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, 0xff, _, _,    0xff, 0xff, _,
/* 1- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
/* 2- */ 0xff,   0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* 3- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
/* 4- */ 0xff,   0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* 5- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
/* 6- */ 0xff,   0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* 7- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
/* 8- */ _,      0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, _,    0xff, _, 0xff, 0xff, 0xff, _,
/* 9- */ 0xffff, 0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, _,    0xff, _,    _,
/* a- */ 0xff,   0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* b- */ 0xffff, 0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* c- */ 0xff,   0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* d- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
/* e- */ 0xff,   0xff, _,    _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _, 0xff, 0xff, 0xff, _,
/* f- */ 0xffff, 0xff, _,    _, _,    0xff, 0xff, _, 0xff, 0xff, _,    _, _,    0xff, 0xff, _,
};
#undef _
// clang-format on

void create_bin_add_fn(Context const& c, llvm::Value* lhs, llvm::Value* rhs)
{
    auto* const lhs_ze = c.builder->CreateZExt(lhs, int_type<uint16_t>(c));
    auto* const rhs_ze = c.builder->CreateZExt(rhs, int_type<uint16_t>(c));
    auto* const lhs_se = c.builder->CreateSExt(lhs, int_type<int16_t>(c));
    auto* const rhs_se = c.builder->CreateSExt(rhs, int_type<int16_t>(c));
    auto* const carry = load_flag(c, C);
    auto* const carry_ze = c.builder->CreateZExt(carry, int_type<uint16_t>(c));

    auto* const result_ze_tmp = c.builder->CreateAdd(lhs_ze, rhs_ze);
    auto* const result_ze = c.builder->CreateAdd(result_ze_tmp, carry_ze);

    auto* const result_se_tmp = c.builder->CreateAdd(lhs_se, rhs_se);
    auto* const result_se = c.builder->CreateAdd(result_se_tmp, carry_ze);

    auto* const gt_127 =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_SGT, result_se, int_const(c, 127_sw));
    auto* const lt_n128 =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_SLT, result_se, int_const(c, -128_sw));
    auto* const new_v = c.builder->CreateOr(gt_127, lt_n128);
    store_flag(c, V, new_v);

    auto* const new_c =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_UGT, result_ze, int_const(c, 255_w));
    store_flag(c, C, new_c);

    auto const result = c.builder->CreateTrunc(result_ze, int_type<uint8_t>(c));

    set_n_z(c, result);
    store_reg(c, A, result);

    c.builder->CreateRetVoid();
}

void create_dec_add_fn(Context const& c, llvm::Value* lhs, llvm::Value* rhs)
{
    auto const lhs_lo = c.builder->CreateAnd(lhs, int_const(c, 0x0f_b));
    auto const lhs_hi = c.builder->CreateLShr(lhs, 4);

    auto const rhs_lo = c.builder->CreateAnd(rhs, int_const(c, 0x0f_b));
    auto const rhs_hi = c.builder->CreateLShr(rhs, 4);

    auto const half_dec_add = [&](llvm::Value* a, llvm::Value* b, llvm::Value* c_bit) {
        auto* const c_byte = c.builder->CreateZExt(c_bit, int_type<uint8_t>(c));
        auto* const result_no_carry = c.builder->CreateAdd(a, b);
        auto* const result_tmp = c.builder->CreateAdd(result_no_carry, c_byte);

        auto* const carry_aloca = c.builder->CreateAlloca(int_type<bool>(c));
        store<bool>(c, int_const(c, false), carry_aloca);
        auto* const result_aloca = c.builder->CreateAlloca(int_type<uint8_t>(c));
        store<uint8_t>(c, result_tmp, result_aloca);

        auto* const ge_10 = c.builder->CreateCmp(
            llvm::CmpInst::Predicate::ICMP_UGE, result_tmp, int_const(c, 10_b));

        auto* const block_ge_10 = llvm::BasicBlock::Create(*c.context, "", c.fn);
        auto* const block_lt_10 = llvm::BasicBlock::Create(*c.context, "", c.fn);

        c.builder->CreateCondBr(ge_10, block_ge_10, block_lt_10);

        // If result >= 10_b
        c.builder->SetInsertPoint(block_ge_10);
        auto* const result_m10 = c.builder->CreateSub(result_tmp, int_const(c, 10_b));
        store<uint8_t>(c, result_m10, result_aloca);
        store<bool>(c, int_const(c, true), carry_aloca);
        c.builder->CreateBr(block_lt_10);

        // If result < 10_b, just continue
        c.builder->SetInsertPoint(block_lt_10);
        auto* const carry = load<bool>(c, carry_aloca);
        auto* const result = load<uint8_t>(c, result_aloca);
        return std::make_pair(result, carry);
    };

    auto* const c_flag = load_flag(c, C);
    auto const [result_lo, carry_lo] = half_dec_add(lhs_lo, rhs_lo, c_flag);
    auto const [result_hi, carry_hi] = half_dec_add(lhs_hi, rhs_hi, carry_lo);

    auto* const result_hi_l = c.builder->CreateShl(result_hi, 4);
    auto* const result = c.builder->CreateOr(result_lo, result_hi_l);

    store_flag(c, C, carry_hi);
    set_n_z(c, result);
    store_reg(c, A, result);

    c.builder->CreateRetVoid();
}

void create_dec_sub_fn(Context const& c, llvm::Value* lhs, llvm::Value* rhs)
{
    auto const lhs_lo = c.builder->CreateAnd(lhs, int_const(c, 0x0f_b));
    auto const lhs_hi = c.builder->CreateLShr(lhs, 4);

    auto const rhs_lo = c.builder->CreateAnd(rhs, int_const(c, 0x0f_b));
    auto const rhs_hi = c.builder->CreateLShr(rhs, 4);

    auto const half_dec_sub = [&](llvm::Value* a, llvm::Value* b, llvm::Value* c_bit) {
        auto* const c_bit_comp = c.builder->CreateNot(c_bit);
        auto* const c_byte = c.builder->CreateZExt(c_bit_comp, int_type<uint8_t>(c));
        auto* const result_no_carry = c.builder->CreateSub(a, b);
        auto* const result_tmp = c.builder->CreateSub(result_no_carry, c_byte);

        auto* const carry_aloca = c.builder->CreateAlloca(int_type<bool>(c));
        store<bool>(c, int_const(c, true), carry_aloca);
        auto* const result_aloca = c.builder->CreateAlloca(int_type<uint8_t>(c));
        store<uint8_t>(c, result_tmp, result_aloca);

        auto* const ge_10 = c.builder->CreateCmp(
            llvm::CmpInst::Predicate::ICMP_UGE, result_tmp, int_const(c, 10_b));

        auto* const block_ge_10 = llvm::BasicBlock::Create(*c.context, "", c.fn);
        auto* const block_lt_10 = llvm::BasicBlock::Create(*c.context, "", c.fn);

        c.builder->CreateCondBr(ge_10, block_ge_10, block_lt_10);

        // If result >= 10_b
        c.builder->SetInsertPoint(block_ge_10);
        auto* const result_p10 = c.builder->CreateAdd(result_tmp, int_const(c, 10_b));
        store<uint8_t>(c, result_p10, result_aloca);
        store<bool>(c, int_const(c, false), carry_aloca);
        c.builder->CreateBr(block_lt_10);

        // If result < 10_b, just continue
        c.builder->SetInsertPoint(block_lt_10);
        auto* const carry =
            c.builder->CreateAlignedLoad(int_type<bool>(c), carry_aloca, llvm::Align::Of<bool>());
        auto* const result = c.builder->CreateAlignedLoad(
            int_type<uint8_t>(c), result_aloca, llvm::Align::Of<uint8_t>());
        return std::make_pair(result, carry);
    };

    auto* const c_flag = load_flag(c, C);
    auto const [result_lo, carry_lo] = half_dec_sub(lhs_lo, rhs_lo, c_flag);
    auto const [result_hi, carry_hi] = half_dec_sub(lhs_hi, rhs_hi, carry_lo);

    auto* const result_hi_l = c.builder->CreateShl(result_hi, 4);
    auto* const result = c.builder->CreateOr(result_lo, result_hi_l);

    store_flag(c, C, carry_hi);
    set_n_z(c, result);
    store_reg(c, A, result);

    c.builder->CreateRetVoid();
}

auto* create_add_sub_fn(Context const& c, bool add)
{
    auto* fn_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*c.context),
                                {llvm::PointerType::get(*c.context, 0), int_type<uint8_t>(c)},
                                false);
    auto* fn = llvm::Function::Create(
        fn_type, llvm::Function::LinkageTypes::PrivateLinkage, add ? "add_fn" : "sub_fn", c.module);
    fn->addFnAttr(llvm::Attribute::AttrKind::InlineHint);
    fn->setCallingConv(llvm::CallingConv::Fast);

    Context new_context = c;
    new_context.fn = fn;

    auto* entry_block = llvm::BasicBlock::Create(*c.context, "entry", fn);
    auto* binary_block = llvm::BasicBlock::Create(*c.context, "binary", fn);
    auto* decimal_block = llvm::BasicBlock::Create(*c.context, "decimal", fn);

    c.builder->SetInsertPoint(entry_block);

    auto* const a_reg = load_reg(new_context, A);
    auto* const d_flag = load_flag(new_context, D);
    auto* const not_decimal =
        c.builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ, d_flag, int_const(c, false));
    auto* const arg_comp = c.builder->CreateNot(fn->getArg(1));
    c.builder->CreateCondBr(not_decimal, binary_block, decimal_block);

    if (add) {
        c.builder->SetInsertPoint(binary_block);
        create_bin_add_fn(new_context, a_reg, fn->getArg(1));
        c.builder->SetInsertPoint(decimal_block);
        create_dec_add_fn(new_context, a_reg, fn->getArg(1));
    } else {
        c.builder->SetInsertPoint(binary_block);
        create_bin_add_fn(new_context, a_reg, arg_comp);
        c.builder->SetInsertPoint(decimal_block);
        create_dec_sub_fn(new_context, a_reg, fn->getArg(1));
    }

    return fn;
}

auto* create_cpu_struct_type(llvm::LLVMContext& context)
{
    auto* sr_struct = llvm::StructType::create(context,
                                               llvm::ArrayRef<llvm::Type*>{
                                                   int_type<bool>(context), // N flag
                                                   int_type<bool>(context), // V flag
                                                   int_type<bool>(context), // D flag
                                                   int_type<bool>(context), // I flag
                                                   int_type<bool>(context), // Z flag
                                                   int_type<bool>(context), // C flag
                                               });
    auto* cpu_struct = llvm::StructType::create(context,
                                                llvm::ArrayRef<llvm::Type*>{
                                                    int_type<uint16_t>(context), // PC register
                                                    int_type<uint8_t>(context),  // SP register
                                                    int_type<uint8_t>(context),  // A register
                                                    int_type<uint8_t>(context),  // X register
                                                    int_type<uint8_t>(context),  // Y register
                                                    sr_struct,                   // SR register
                                                },
                                                "emu_CPU");

    return cpu_struct;
}

auto* create_jit_function_type(llvm::LLVMContext& context)
{
    auto* fn_type = llvm::FunctionType::get(int_type<uint64_t>(context),
                                            {llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0)},
                                            false);
    return fn_type;
}

auto* create_instruction_function_type(llvm::LLVMContext& context)
{
    auto* fn_type = llvm::FunctionType::get(int_type<uint64_t>(context),
                                            {llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0)},
                                            false);
    return fn_type;
}

auto* create_addr_mode_function_type(llvm::LLVMContext& context)
{
    auto* fn_type = llvm::FunctionType::get(int_type<uint64_t>(context),
                                            {llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0),
                                             llvm::PointerType::get(context, 0)},
                                            false);
    return fn_type;
}

auto* create_bus_read_function_type(llvm::LLVMContext& context)
{
    auto* fn_type =
        llvm::FunctionType::get(int_type<uint8_t>(context),
                                {llvm::PointerType::get(context, 0), int_type<uint16_t>(context)},
                                false);
    return fn_type;
}

auto* create_bus_write_function_type(llvm::LLVMContext& context)
{
    auto* fn_type = llvm::FunctionType::get(llvm::Type::getVoidTy(context),
                                            {llvm::PointerType::get(context, 0),
                                             int_type<uint16_t>(context),
                                             int_type<uint8_t>(context)},
                                            false);
    return fn_type;
}

auto* create_call_function_function_type(llvm::LLVMContext& context)
{
    auto* fn_type =
        llvm::FunctionType::get(int_type<uint64_t>(context),
                                {llvm::PointerType::get(context, 0), int_type<uint16_t>(context)},
                                false);
    return fn_type;
}

} // namespace

void optimize(llvm::Module& module, llvm::OptimizationLevel opt_level)
{
    // Create the analysis managers.
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    const auto gen_level = [opt_level] {
        auto ret = llvm::CodeGenOpt::getLevel(opt_level.getSpeedupLevel());
        return ret ? *ret : *llvm::CodeGenOpt::getLevel(0);
    }();

    // Create the new pass manager builder.
    // Take a look at the PassBuilder constructor parameters for more
    // customization, e.g. specifying a TargetMachine or various debugging
    // options.
    auto target_machine =
        std::move(*llvm::orc::JITTargetMachineBuilder::detectHost()->createTargetMachine());
    target_machine->setOptLevel(gen_level);
    llvm::PassBuilder PB{target_machine.get()};

    // Register all the basic analyses with the managers.
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Create the pass manager.
    llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(opt_level);

    // Optimize the IR!
    MPM.run(module, MAM);
}

void build_base()
{
    auto context = std::make_unique<llvm::LLVMContext>();
    auto module = std::make_unique<llvm::Module>("base_functions", *context);
    auto builder = std::make_unique<llvm::IRBuilder<>>(*context);

#ifndef NDEBUG
    auto* cpu_struct = create_cpu_struct_type(*context);

    // Sanity check
    auto* cpu_struct_layout = module->getDataLayout().getStructLayout(cpu_struct);
    auto* sr_struct_layout = module->getDataLayout().getStructLayout(static_cast<llvm::StructType*>(
        cpu_struct->getElementType(std::to_underlying(RegOffset::SR))));
    assert(cpu_struct_layout->getSizeInBytes() == sizeof(CPU));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(PC)) == offsetof(CPU, PC));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(SP)) == offsetof(CPU, SP));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(A)) == offsetof(CPU, A));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(X)) == offsetof(CPU, X));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(Y)) == offsetof(CPU, Y));
    assert(cpu_struct_layout->getElementOffset(std::to_underlying(RegOffset::SR))
           == offsetof(CPU, SR));

    assert(sr_struct_layout->getElementOffset(std::to_underlying(N))
           == offsetof(decltype(CPU::SR), N));
    assert(sr_struct_layout->getElementOffset(std::to_underlying(V))
           == offsetof(decltype(CPU::SR), V));
    assert(sr_struct_layout->getElementOffset(std::to_underlying(D))
           == offsetof(decltype(CPU::SR), D));
    assert(sr_struct_layout->getElementOffset(std::to_underlying(I))
           == offsetof(decltype(CPU::SR), I));
    assert(sr_struct_layout->getElementOffset(std::to_underlying(Z))
           == offsetof(decltype(CPU::SR), Z));
    assert(sr_struct_layout->getElementOffset(std::to_underlying(C))
           == offsetof(decltype(CPU::SR), C));
#endif // NDEBUG
}

std::unique_ptr<llvm::Module> codegen(llvm::orc::ThreadSafeContext tsc,
                                      std::map<word_t, control_block> const& flow)
{
    auto lock = tsc.getLock();
    auto* context = tsc.getContext();
    auto module = std::make_unique<llvm::Module>("test_module", *context);
    auto builder = std::make_unique<llvm::IRBuilder<>>(*context);

    auto* cpu_struct = create_cpu_struct_type(*context);

    auto* fn_type = create_jit_function_type(*context);
    auto* fn = llvm::Function::Create(fn_type,
                                      llvm::Function::LinkageTypes::ExternalLinkage,
                                      fmt::format("fn_{:04x}", flow.begin()->first),
                                      module.get());

    auto* read_bus_fn_type = create_bus_read_function_type(*context);
    auto* read_bus_fn = llvm::Function::Create(
        read_bus_fn_type, llvm::Function::LinkageTypes::ExternalLinkage, "read_bus", module.get());

    auto* write_bus_fn_type = create_bus_write_function_type(*context);
    auto* write_bus_fn = llvm::Function::Create(write_bus_fn_type,
                                                llvm::Function::LinkageTypes::ExternalLinkage,
                                                "write_bus",
                                                module.get());

    auto* call_function_fn_type = create_call_function_function_type(*context);
    auto* call_function_fn = llvm::Function::Create(call_function_fn_type,
                                                    llvm::Function::LinkageTypes::ExternalLinkage,
                                                    "call_function",
                                                    module.get());

    auto* add_fn = create_add_sub_fn(
        {
            .context = context,
            .builder = builder.get(),
            .cpu_type = cpu_struct,
            .module = module.get(),
        },
        true);

    auto* sub_fn = create_add_sub_fn(
        {
            .context = context,
            .builder = builder.get(),
            .cpu_type = cpu_struct,
            .module = module.get(),
        },
        false);

    auto* const entry_block = llvm::BasicBlock::Create(*context, "", fn);
    builder->SetInsertPoint(entry_block);
    auto* const cycle_counter = builder->CreateAlloca(int_type<uint64_t>(*context));
    builder->CreateAlignedStore(
        int_const<uint64_t>(*context, 0), cycle_counter, llvm::Align::Of<uint64_t>());
    auto* const aux_8b = builder->CreateAlloca(int_type<uint8_t>(*context));

    std::unordered_map<word_t, llvm::BasicBlock*> blocks;
    for (auto const& entry : flow) {
        blocks.emplace(
            entry.first,
            llvm::BasicBlock::Create(*context, fmt::format("label_{:04x}", entry.first), fn));
    }

    auto first = blocks.at(flow.begin()->first);
    builder->CreateBr(first);

    Context ctx{
        .context = context,
        .builder = builder.get(),
        .cpu_type = cpu_struct,
        .fn = fn,
        .read_bus_fn = read_bus_fn,
        .write_bus_fn = write_bus_fn,
        .add_fn = add_fn,
        .sub_fn = sub_fn,
        .call_function_fn = call_function_fn,
        .cycle_counter_ptr = cycle_counter,
        .aux_8b_ptr = aux_8b,
    };

    for (auto const& entry : flow) {
        auto* const block = blocks.at(entry.first);
        builder->SetInsertPoint(block);
        llvm::Value* last_result = nullptr;
        for (auto const& instruction : entry.second.instructions) {
            ctx.inst = &instruction;
            auto* const codegen_fn = instruction_table[static_cast<size_t>(instruction.bytes[0])];

            // Check for modification of expected instruction. If modified, abort and let the
            // interpreter continue the execution.
            auto const inst_repr = instruction.get_16bit_representation();
            auto* const p =
                builder->CreateGEP(llvm::ArrayType::get(int_type<uint8_t>(ctx), 0x10000),
                                   ctx.fn->getArg(2),
                                   {int_const(ctx, 0), int_const(ctx, instruction.pc)});
            auto* const load_inst_repr = unaligned_load<uint16_t>(ctx, p);
            auto const mask = word_t{self_mod_table[static_cast<size_t>(instruction.bytes[0])]};
            auto* const masked_inst_repr = builder->CreateAnd(load_inst_repr, int_const(ctx, mask));
            auto* const inst_matches = builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_EQ,
                                                          masked_inst_repr,
                                                          int_const(ctx, inst_repr & mask));

            auto* const inst_matches_block = llvm::BasicBlock::Create(*context, "", fn);
            auto* const inst_doesnt_match_block = llvm::BasicBlock::Create(*context, "", fn);

            builder->CreateCondBr(
                expect(ctx, inst_matches, true), inst_matches_block, inst_doesnt_match_block);

            // If the instruction doesn't match, return
            builder->SetInsertPoint(inst_doesnt_match_block);
            return_function(ctx, int_const(ctx, instruction.pc));

            // Else, just continue normally.
            builder->SetInsertPoint(inst_matches_block);

            last_result = codegen_fn(ctx);
            auto const next_addr = instruction.get_pc() + word_t{instruction.length};

            if (instruction.is_call()) {
                make_function_call(ctx, last_result, next_addr);
            }
        }

        if (entry.second.next_taken && entry.second.next_not_taken) {
            // Conditional branch
            auto* const taken_block = blocks.at(*entry.second.next_taken);
            auto* const not_taken_block = blocks.at(*entry.second.next_not_taken);
            builder->CreateCondBr(last_result, taken_block, not_taken_block);
        } else if (!entry.second.next_taken && entry.second.next_not_taken) {
            // Continue to next block
            auto* const not_taken_block = blocks.at(*entry.second.next_not_taken);
            builder->CreateBr(not_taken_block);
        } else if (entry.second.next_taken && !entry.second.next_not_taken) {
            // Inconditional branch
            auto* const taken_block = blocks.at(*entry.second.next_taken);
            builder->CreateBr(taken_block);
        } else {
            // Termination block, return.
            return_function(ctx, last_result);
        }
    }

    if (llvm::verifyFunction(*fn, &llvm::errs())) {
        module->print(llvm::errs(), nullptr);
        return nullptr;
    }

    return module;
}

} // namespace emu