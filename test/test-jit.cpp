#include "../src/jit.h"
#include "../src/codegen.h"
#include "../src/control_flow.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>

using namespace emu::literals;

namespace {
llvm::ExitOnError exit_on_error;
}

struct MemData {
    emu::word_t offset{};
    std::vector<emu::byte_t> data{};
};

struct ParamType {
    std::string name;
    std::vector<emu::byte_t> instructions;
    std::vector<MemData> data{};
    uint64_t expected_cycles;
    emu::word_t fn_address{0x55f0_w};
    emu::word_t return_address{0x1234_w};
    emu::CPU initial_cpu_state{};
    bool do_rti{};
    emu::CPU expected_cpu_state{};
};

std::ostream& operator<<(std::ostream& os, ParamType const& p)
{
    os << p.name;
    return os;
}

class TestJitCodegen
    : public ::testing::Test
    , public ::testing::WithParamInterface<ParamType> {
protected:
    // You can remove any or all of the following functions if their bodies would be empty.

    TestJitCodegen()
        : _context{std::make_unique<llvm::LLVMContext>()}
        , _jit{exit_on_error(emu::make_jit())}
    {}

    // If the constructor and destructor are not enough for setting up and cleaning up each test,
    // you can define the following methods:

    void SetUp() override
    {
        auto const& params = this->GetParam();

        _cpu = params.initial_cpu_state;
        _cpu.PC = params.fn_address,
        std::ranges::copy(params.instructions,
                          _bus.memory_space.begin() + static_cast<size_t>(_cpu.PC));
        for (auto const& data : params.data) {
            std::ranges::copy(data.data,
                              _bus.memory_space.begin() + static_cast<size_t>(data.offset));
        }

        // Add return instruction
        emu::byte_t const ret_instruction = params.do_rti ? 0x40_b : 0x60_b;
        _bus.memory_space[static_cast<size_t>(_cpu.PC) + params.instructions.size()] =
            ret_instruction;

        // Set up stack
        _bus.write(0x1ff_w, emu::get_hi(params.return_address));
        _bus.write(0x1fe_w, emu::get_lo(params.return_address));

        // Add SR to the stack if the return instruction is an RTI
        if (params.do_rti) {
            _bus.write(0x1fd_w, static_cast<emu::byte_t>(params.expected_cpu_state.SR));
            _cpu.SP = 0xfc_b;
        } else {
            _cpu.SP = 0xfd_b;
        }
    }

    std::string get_fn_name() const { return fmt::format("fn_{:04x}", _cpu.PC); }

    static bool test_error(llvm::Error error)
    {
        if (error) {
            llvm::logAllUnhandledErrors(std::move(error), llvm::errs());
            return true;
        } else {
            return false;
        }
    }

    template<typename T>
    static bool test_expected(llvm::Expected<T>& exp)
    {
        if (exp) {
            return true;
        } else {
            llvm::logAllUnhandledErrors(exp.takeError(), llvm::errs());
            return false;
        }
    }

    llvm::orc::ThreadSafeContext _context;
    std::unique_ptr<llvm::orc::LLJIT> _jit;
    emu::CPU _cpu{};
    emu::Bus _bus{};
};

TEST_P(TestJitCodegen, Test)
{
    auto const& params = this->GetParam();
    auto const control_flow = emu::build_control_flow(_bus, _cpu.PC);
    auto module = emu::codegen(_context, control_flow);
    ASSERT_TRUE(module);
    auto err = emu::materialize(*_jit, _context, std::move(module));
    ASSERT_FALSE(test_error(std::move(err)));

    auto fn_ex = _jit->lookup(get_fn_name());
    ASSERT_TRUE(test_expected(fn_ex));
    auto* const fn = fn_ex.get().toPtr<emu::jit_fn_t>();

    auto const cycles = fn(_cpu, _bus);

    EXPECT_EQ(cycles, params.expected_cycles + 6);

    auto expected_cpu = params.expected_cpu_state;
    expected_cpu.SP = 0xff_b;
    expected_cpu.PC = params.return_address;

    EXPECT_EQ(_cpu, expected_cpu);
}

static std::vector<ParamType> get_test_cases()
{
    std::vector<ParamType> ret = {
        {
            .name = "Test RTS",
            .instructions = {},
            .expected_cycles = 0,
        },
        {
            .name = "Test RTI",
            .instructions = {},
            .expected_cycles = 0,
            .do_rti = true,
            .expected_cpu_state = {.SR = 0xff_b},
        },
        {
            .name = "Test NOP",
            .instructions = {0xea_b},
            .expected_cycles = 2,
        },
    };

    auto const add_logic_instruction_tests = [&ret](std::string_view name,
                                                    std::string_view condition,
                                                    std::vector<emu::byte_t> const& opcodes,
                                                    emu::byte_t param_a,
                                                    emu::byte_t param_b,
                                                    emu::byte_t result,
                                                    emu::SR expected_flags = {}) {
        auto const op_imm = opcodes[0];
        auto const op_zpg = opcodes[1];
        auto const op_zpg_X = opcodes[2];
        auto const op_abs = opcodes[3];
        auto const op_abs_X = opcodes[4];
        auto const op_abs_Y = opcodes[5];
        auto const op_X_ind = opcodes[6];
        auto const op_ind_Y = opcodes[7];

        ret.insert(
            ret.end(),
            {
                {
                    .name = fmt::format("Test {} imm, {}", name, condition),
                    .instructions = {op_imm, param_b},
                    .expected_cycles = 2,
                    .initial_cpu_state = {.A = param_a},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg, {}", name, condition),
                    .instructions = {op_zpg, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {param_b}}},
                    .expected_cycles = 3,
                    .initial_cpu_state = {.A = param_a},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg,X - no overflow, {}", name, condition),
                    .instructions = {op_zpg_X, 0x55_b},
                    .data = {{.offset = 0x65_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0x10_b},
                    .expected_cpu_state = {.A = result, .X = 0x10_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg,X - overflow, {}", name, condition),
                    .instructions = {op_zpg_X, 0x55_b},
                    .data = {{.offset = 0x45_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0xf0_b},
                    .expected_cpu_state = {.A = result, .X = 0xf0_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} absolute, {}", name, condition),
                    .instructions = {op_abs, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,X - no page boundary cross, {}", name, condition),
                    .instructions = {op_abs_X, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0x20_b},
                    .expected_cpu_state = {.A = result, .X = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,X - page boundary cross, {}", name, condition),
                    .instructions = {op_abs_X, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .X = 0x50_b},
                    .expected_cpu_state = {.A = result, .X = 0x50_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,Y - no page boundary cross, {}", name, condition),
                    .instructions = {op_abs_Y, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .Y = 0x20_b},
                    .expected_cpu_state = {.A = result, .Y = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,Y - page boundary cross, {}", name, condition),
                    .instructions = {op_abs_Y, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .Y = 0x50_b},
                    .expected_cpu_state = {.A = result, .Y = 0x50_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect,X) - no overflow, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0x75_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0x20_b},
                    .expected_cpu_state = {.A = result, .X = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect,X) - overflow, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0x45_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0xf0_b},
                    .expected_cpu_state = {.A = result, .X = 0xf0_b, .SR = expected_flags},
                },
                {
                    .name =
                        fmt::format("Test {} (indirect,X) - overflow at edge, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0xff_w, .data = {0xbc_b}},
                             {.offset = 0x00_w, .data = {0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0xaa_b},
                    .expected_cpu_state = {.A = result, .X = 0xaa_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect),Y - no overflow, {}", name, condition),
                    .instructions = {op_ind_Y, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .Y = 0x20_b},
                    .expected_cpu_state = {.A = result, .Y = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect),Y - overflow, {}", name, condition),
                    .instructions = {op_ind_Y, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .Y = 0x50_b},
                    .expected_cpu_state = {.A = result, .Y = 0x50_b, .SR = expected_flags},
                },
            });
    };

    auto const add_conditional_branch_instruction_tests = [&ret](std::string_view name,
                                                                 emu::byte_t opcode,
                                                                 emu::SR taken_condition,
                                                                 emu::SR not_taken_condition) {
        ret.insert(ret.end(),
                   {
                       {
                           .name = fmt::format("Test {} - not taken positive, same page", name),
                           .instructions = {opcode, 0x02_b, 0xea_b, 0xea_b},
                           .expected_cycles = 6,
                           .initial_cpu_state = {.SR = not_taken_condition},
                           .expected_cpu_state = {.SR = not_taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - taken positive, same page", name),
                           .instructions = {opcode, 0x02_b, 0xea_b, 0xea_b},
                           .expected_cycles = 3,
                           .initial_cpu_state = {.SR = taken_condition},
                           .expected_cpu_state = {.SR = taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - not taken positive, next page", name),
                           .instructions = {opcode, 0x02_b, 0xea_b, 0xea_b},
                           .expected_cycles = 6,
                           .fn_address = 0x55fe_w,
                           .initial_cpu_state = {.SR = not_taken_condition},
                           .expected_cpu_state = {.SR = not_taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - taken positive, next page", name),
                           .instructions = {opcode, 0x02_b, 0xea_b, 0xea_b},
                           .expected_cycles = 4,
                           .fn_address = 0x55fe_w,
                           .initial_cpu_state = {.SR = taken_condition},
                           .expected_cpu_state = {.SR = taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - not taken negative, same page", name),
                           .instructions =
                               {0x4c_b, 0xf4_b, 0x55_b, 0x60_b, opcode, 0xfd_b, 0xea_b, 0xea_b},
                           .expected_cycles = 9,
                           .initial_cpu_state = {.SR = not_taken_condition},
                           .expected_cpu_state = {.SR = not_taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - taken negative, same page", name),
                           .instructions =
                               {0x4c_b, 0xf4_b, 0x55_b, 0x60_b, opcode, 0xfd_b, 0xea_b, 0xea_b},
                           .expected_cycles = 6,
                           .initial_cpu_state = {.SR = taken_condition},
                           .expected_cpu_state = {.SR = taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - not taken negative, previous page", name),
                           .instructions =
                               {0x4c_b, 0x00_b, 0x56_b, 0x60_b, opcode, 0xfd_b, 0xea_b, 0xea_b},
                           .expected_cycles = 9,
                           .fn_address = 0x55fc_w,
                           .initial_cpu_state = {.SR = not_taken_condition},
                           .expected_cpu_state = {.SR = not_taken_condition},
                       },
                       {
                           .name = fmt::format("Test {} - taken negative, previous page", name),
                           .instructions =
                               {0x4c_b, 0x00_b, 0x56_b, 0x60_b, opcode, 0xfd_b, 0xea_b, 0xea_b},
                           .expected_cycles = 7,
                           .fn_address = 0x55fc_w,
                           .initial_cpu_state = {.SR = taken_condition},
                           .expected_cpu_state = {.SR = taken_condition},
                       },
                   });
    };

    std::vector const and_opcodes{0x29_b, 0x25_b, 0x35_b, 0x2d_b, 0x3d_b, 0x39_b, 0x21_b, 0x31_b};
    std::vector const ora_opcodes{0x09_b, 0x05_b, 0x15_b, 0x0d_b, 0x1d_b, 0x19_b, 0x01_b, 0x11_b};
    std::vector const eor_opcodes{0x49_b, 0x45_b, 0x55_b, 0x4d_b, 0x5d_b, 0x59_b, 0x41_b, 0x51_b};

    add_logic_instruction_tests("AND", "no flags", and_opcodes, 0xfc_b, 0x3f_b, 0x3c_b);
    add_logic_instruction_tests("AND", "Z flag", and_opcodes, 0x55_b, 0xaa_b, 0x00_b, {.Z = true});
    add_logic_instruction_tests("AND", "N flag", and_opcodes, 0xd5_b, 0xaa_b, 0x80_b, {.N = true});

    add_logic_instruction_tests("ORA", "no flags", ora_opcodes, 0x54_b, 0x2c_b, 0x7c_b);
    add_logic_instruction_tests("ORA", "Z flag", ora_opcodes, 0x00_b, 0x00_b, 0x00_b, {.Z = true});
    add_logic_instruction_tests("ORA", "N flag", ora_opcodes, 0xd4_b, 0x2c_b, 0xfc_b, {.N = true});

    add_logic_instruction_tests("EOR", "no flags", eor_opcodes, 0x6c_b, 0x36_b, 0x5a_b);
    add_logic_instruction_tests("EOR", "Z flag", eor_opcodes, 0xaa_b, 0xaa_b, 0x00_b, {.Z = true});
    add_logic_instruction_tests("EOR", "N flag", eor_opcodes, 0x6c_b, 0xb6_b, 0xda_b, {.N = true});

    add_conditional_branch_instruction_tests("BCC", 0x90_b, {}, {.C = true});
    add_conditional_branch_instruction_tests("BCS", 0xb0_b, {.C = true}, {});
    add_conditional_branch_instruction_tests("BEQ", 0xf0_b, {.Z = true}, {});
    add_conditional_branch_instruction_tests("BMI", 0x30_b, {.N = true}, {});
    add_conditional_branch_instruction_tests("BNE", 0xd0_b, {}, {.Z = true});
    add_conditional_branch_instruction_tests("BPL", 0x10_b, {}, {.N = true});
    add_conditional_branch_instruction_tests("BVC", 0x50_b, {}, {.V = true});
    add_conditional_branch_instruction_tests("BVS", 0x70_b, {.V = true}, {});

    return ret;
}

INSTANTIATE_TEST_SUITE_P(Test,
                         TestJitCodegen,
                         ::testing::ValuesIn(get_test_cases()),
                         [](testing::TestParamInfo<ParamType> const& info) {
                             auto ret = info.param.name;
                             std::ranges::transform(ret, ret.begin(), [](char c) {
                                 if (c >= 'a' && c <= 'z') return c;
                                 if (c >= 'A' && c <= 'Z') return c;
                                 if (c >= '0' && c <= '9') return c;
                                 return '_';
                             });
                             return ret;
                         });
