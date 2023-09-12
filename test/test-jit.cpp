#include "../src/jit.h"
#include "../src/codegen.h"
#include "../src/control_flow.h"
#include "../src/emulator.h"

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
    std::vector<MemData> expected_data{};
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
        if (params.do_rti) {
            _bus.write(0x1ff_w, emu::get_hi(params.return_address));
            _bus.write(0x1fe_w, emu::get_lo(params.return_address));
            _bus.write(0x1fd_w, static_cast<emu::byte_t>(params.expected_cpu_state.SR));
            _cpu.SP = 0xfc_b;
        } else {
            _bus.write(0x1ff_w, emu::get_hi(params.return_address - 1_w));
            _bus.write(0x1fe_w, emu::get_lo(params.return_address - 1_w));
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
    emu::Emulator _em{};
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

    auto const cycles = fn(_cpu, _bus, _em);

    EXPECT_EQ(cycles, params.expected_cycles + 6);

    auto expected_cpu = params.expected_cpu_state;
    expected_cpu.SP = 0xff_b;
    expected_cpu.PC = params.return_address;

    EXPECT_EQ(_cpu, expected_cpu);

    for (auto const& data : params.expected_data) {
        EXPECT_TRUE(std::equal(data.data.cbegin(),
                               data.data.cend(),
                               _bus.memory_space.cbegin() + static_cast<size_t>(data.offset)));
    }
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
            .name = "Test RTI alternate flags",
            .instructions = {},
            .expected_cycles = 0,
            .do_rti = true,
            .expected_cpu_state = {.SR = 0xba_b},
        },
        {
            .name = "Test NOP",
            .instructions = {0xea_b},
            .expected_cycles = 2,
        },
        {
            .name = "Test STA zpg",
            .instructions = {0x85_b, 0xaa_b},
            .expected_data = {{.offset = 0xaa_w, .data = {0x55_b}}},
            .expected_cycles = 3,
            .initial_cpu_state = {.A = 0x55_b},
            .expected_cpu_state = {.A = 0x55_b},
        },
        {
            .name = "Test STA zpg,X",
            .instructions = {0x95_b, 0xaa_b},
            .expected_data = {{.offset = 0xbb_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.A = 0x55_b, .X = 0x11_b},
            .expected_cpu_state = {.A = 0x55_b, .X = 0x11_b},
        },
        {
            .name = "Test STA abs",
            .instructions = {0x8d_b, 0xaa_b, 0x55_b},
            .expected_data = {{.offset = 0x55aa_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.A = 0x55_b},
            .expected_cpu_state = {.A = 0x55_b},
        },
        {
            .name = "Test STA abs,X",
            .instructions = {0x9d_b, 0xaa_b, 0x55_b},
            .expected_data = {{.offset = 0x55bb_w, .data = {0x55_b}}},
            .expected_cycles = 5,
            .initial_cpu_state = {.A = 0x55_b, .X = 0x11_b},
            .expected_cpu_state = {.A = 0x55_b, .X = 0x11_b},
        },
        {
            .name = "Test STA abs,Y",
            .instructions = {0x99_b, 0xaa_b, 0x55_b},
            .expected_data = {{.offset = 0x55bb_w, .data = {0x55_b}}},
            .expected_cycles = 5,
            .initial_cpu_state = {.A = 0x55_b, .Y = 0x11_b},
            .expected_cpu_state = {.A = 0x55_b, .Y = 0x11_b},
        },
        {
            .name = "Test STA (indirect,X)",
            .instructions = {0x81_b, 0xaa_b},
            .data = {{.offset = 0xbb_w, .data = {0xaa_b, 0x55_b}}},
            .expected_data = {{.offset = 0x55aa_w, .data = {0x55_b}}},
            .expected_cycles = 6,
            .initial_cpu_state = {.A = 0x55_b, .X = 0x11_b},
            .expected_cpu_state = {.A = 0x55_b, .X = 0x11_b},
        },
        {
            .name = "Test STA (indirect),Y",
            .instructions = {0x91_b, 0xaa_b},
            .data = {{.offset = 0xaa_w, .data = {0xaa_b, 0x55_b}}},
            .expected_data = {{.offset = 0x55bb_w, .data = {0x55_b}}},
            .expected_cycles = 6,
            .initial_cpu_state = {.A = 0x55_b, .Y = 0x11_b},
            .expected_cpu_state = {.A = 0x55_b, .Y = 0x11_b},
        },
        {
            .name = "Test STX zpg",
            .instructions = {0x86_b, 0xaa_b},
            .expected_data = {{.offset = 0xaa_w, .data = {0x55_b}}},
            .expected_cycles = 3,
            .initial_cpu_state = {.X = 0x55_b},
            .expected_cpu_state = {.X = 0x55_b},
        },
        {
            .name = "Test STX zpg,Y",
            .instructions = {0x96_b, 0xaa_b},
            .expected_data = {{.offset = 0xbb_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.X = 0x55_b, .Y = 0x11_b},
            .expected_cpu_state = {.X = 0x55_b, .Y = 0x11_b},
        },
        {
            .name = "Test STX abs",
            .instructions = {0x8e_b, 0xaa_b, 0x55_b},
            .expected_data = {{.offset = 0x55aa_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.X = 0x55_b},
            .expected_cpu_state = {.X = 0x55_b},
        },
        {
            .name = "Test STY zpg",
            .instructions = {0x84_b, 0xaa_b},
            .expected_data = {{.offset = 0xaa_w, .data = {0x55_b}}},
            .expected_cycles = 3,
            .initial_cpu_state = {.Y = 0x55_b},
            .expected_cpu_state = {.Y = 0x55_b},
        },
        {
            .name = "Test STY zpg,X",
            .instructions = {0x94_b, 0xaa_b},
            .expected_data = {{.offset = 0xbb_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.X = 0x11_b, .Y = 0x55_b},
            .expected_cpu_state = {.X = 0x11_b, .Y = 0x55_b},
        },
        {
            .name = "Test STY abs",
            .instructions = {0x8c_b, 0xaa_b, 0x55_b},
            .expected_data = {{.offset = 0x55aa_w, .data = {0x55_b}}},
            .expected_cycles = 4,
            .initial_cpu_state = {.Y = 0x55_b},
            .expected_cpu_state = {.Y = 0x55_b},
        },
    };

    auto const add_group_one_tests = [&ret](std::string_view name,
                                            std::string_view condition,
                                            std::array<emu::byte_t, 8> const& opcodes,
                                            emu::byte_t param_a,
                                            emu::byte_t param_b,
                                            emu::byte_t result,
                                            emu::SR initial_flags = {},
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
                    .initial_cpu_state = {.A = param_a, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg, {}", name, condition),
                    .instructions = {op_zpg, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {param_b}}},
                    .expected_cycles = 3,
                    .initial_cpu_state = {.A = param_a, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg,X - no addr ovf, {}", name, condition),
                    .instructions = {op_zpg_X, 0x55_b},
                    .data = {{.offset = 0x65_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0x10_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0x10_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} zpg,X - addr ovf, {}", name, condition),
                    .instructions = {op_zpg_X, 0x55_b},
                    .data = {{.offset = 0x45_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0xf0_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0xf0_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} absolute, {}", name, condition),
                    .instructions = {op_abs, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,X - no page boundary cross, {}", name, condition),
                    .instructions = {op_abs_X, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .X = 0x20_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,X - page boundary cross, {}", name, condition),
                    .instructions = {op_abs_X, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .X = 0x50_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0x50_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,Y - no page boundary cross, {}", name, condition),
                    .instructions = {op_abs_Y, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 4,
                    .initial_cpu_state = {.A = param_a, .Y = 0x20_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .Y = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format(
                        "Test {} absolute,Y - page boundary cross, {}", name, condition),
                    .instructions = {op_abs_Y, 0xbc_b, 0x9a_b},
                    .data = {{.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .Y = 0x50_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .Y = 0x50_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect,X) - no addr ovf, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0x75_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0x20_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect,X) - addr ovf, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0x45_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0xf0_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0xf0_b, .SR = expected_flags},
                },
                {
                    .name =
                        fmt::format("Test {} (indirect,X) - addr ovf at edge, {}", name, condition),
                    .instructions = {op_X_ind, 0x55_b},
                    .data = {{.offset = 0xff_w, .data = {0xbc_b}},
                             {.offset = 0x00_w, .data = {0x9a_b}},
                             {.offset = 0x9abc_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .X = 0xaa_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .X = 0xaa_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect),Y - no addr ovf, {}", name, condition),
                    .instructions = {op_ind_Y, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9adc_w, .data = {param_b}}},
                    .expected_cycles = 5,
                    .initial_cpu_state = {.A = param_a, .Y = 0x20_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .Y = 0x20_b, .SR = expected_flags},
                },
                {
                    .name = fmt::format("Test {} (indirect),Y - addr ovf, {}", name, condition),
                    .instructions = {op_ind_Y, 0x55_b},
                    .data = {{.offset = 0x55_w, .data = {0xbc_b, 0x9a_b}},
                             {.offset = 0x9b0c_w, .data = {param_b}}},
                    .expected_cycles = 6,
                    .initial_cpu_state = {.A = param_a, .Y = 0x50_b, .SR = initial_flags},
                    .expected_cpu_state = {.A = result, .Y = 0x50_b, .SR = expected_flags},
                },
            });
    };

    auto const add_cond_branch_inst_tests = [&ret](std::string_view name,
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

    auto const add_shift_inst_test = [&ret](std::string_view name,
                                            std::string_view condition,
                                            std::array<emu::byte_t, 5> const& opcodes,
                                            emu::byte_t operand,
                                            emu::byte_t result,
                                            emu::SR initial_sr = {},
                                            emu::SR expected_sr = {}) {
        auto const op_acc = opcodes[0];
        auto const op_zpg = opcodes[1];
        auto const op_zpg_X = opcodes[2];
        auto const op_abs = opcodes[3];
        auto const op_abs_X = opcodes[4];
        ret.insert(ret.end(),
                   {
                       {
                           .name = fmt::format("Test {} - accumulator - {}", name, condition),
                           .instructions = {op_acc},
                           .expected_cycles = 2,
                           .initial_cpu_state = {.A = operand, .SR = initial_sr},
                           .expected_cpu_state = {.A = result, .SR = expected_sr},
                       },
                       {
                           .name = fmt::format("Test {} - zpg - {}", name, condition),
                           .instructions = {op_zpg, 0x55_b},
                           .data = {{.offset = 0x55_w, .data = {operand}}},
                           .expected_data = {{.offset = 0x55_w, .data = {result}}},
                           .expected_cycles = 5,
                           .initial_cpu_state = {.SR = initial_sr},
                           .expected_cpu_state = {.SR = expected_sr},
                       },
                       {
                           .name = fmt::format("Test {} - zpg,X - {}", name, condition),
                           .instructions = {op_zpg_X, 0x55_b},
                           .data = {{.offset = 0x66_w, .data = {operand}}},
                           .expected_data = {{.offset = 0x66_w, .data = {result}}},
                           .expected_cycles = 6,
                           .initial_cpu_state = {.X = 0x11_b, .SR = initial_sr},
                           .expected_cpu_state = {.X = 0x11_b, .SR = expected_sr},
                       },
                       {
                           .name = fmt::format("Test {} - absolute - {}", name, condition),
                           .instructions = {op_abs, 0x55_b, 0xaa_b},
                           .data = {{.offset = 0xaa55_w, .data = {operand}}},
                           .expected_data = {{.offset = 0xaa55_w, .data = {result}}},
                           .expected_cycles = 6,
                           .initial_cpu_state = {.SR = initial_sr},
                           .expected_cpu_state = {.SR = expected_sr},
                       },
                       {
                           .name = fmt::format("Test {} - absolute,X - {}", name, condition),
                           .instructions = {op_abs_X, 0x55_b, 0xaa_b},
                           .data = {{.offset = 0xaa66_w, .data = {operand}}},
                           .expected_data = {{.offset = 0xaa66_w, .data = {result}}},
                           .expected_cycles = 7,
                           .initial_cpu_state = {.X = 0x11_b, .SR = initial_sr},
                           .expected_cpu_state = {.X = 0x11_b, .SR = expected_sr},
                       },
                   });
    };

    std::array const and_opcodes{0x29_b, 0x25_b, 0x35_b, 0x2d_b, 0x3d_b, 0x39_b, 0x21_b, 0x31_b};
    std::array const ora_opcodes{0x09_b, 0x05_b, 0x15_b, 0x0d_b, 0x1d_b, 0x19_b, 0x01_b, 0x11_b};
    std::array const eor_opcodes{0x49_b, 0x45_b, 0x55_b, 0x4d_b, 0x5d_b, 0x59_b, 0x41_b, 0x51_b};
    std::array const lda_opcodes{0xa9_b, 0xa5_b, 0xb5_b, 0xad_b, 0xbd_b, 0xb9_b, 0xa1_b, 0xb1_b};
    std::array const adc_opcodes{0x69_b, 0x65_b, 0x75_b, 0x6d_b, 0x7d_b, 0x79_b, 0x61_b, 0x71_b};
    std::array const sbc_opcodes{0xe9_b, 0xe5_b, 0xf5_b, 0xed_b, 0xfd_b, 0xf9_b, 0xe1_b, 0xf1_b};
    std::array const asl_opcodes{0x0a_b, 0x06_b, 0x16_b, 0x0e_b, 0x1e_b};
    std::array const lsr_opcodes{0x4a_b, 0x46_b, 0x56_b, 0x4e_b, 0x5e_b};
    std::array const rol_opcodes{0x2a_b, 0x26_b, 0x36_b, 0x2e_b, 0x3e_b};
    std::array const ror_opcodes{0x6a_b, 0x66_b, 0x76_b, 0x6e_b, 0x7e_b};

    // clang-format off
    #define Z .Z=true
    #define N .N=true
    #define C .C=true
    #define V .V=true
    #define D .D=true

    add_group_one_tests("AND", "no flags", and_opcodes, 0xfc_b, 0x3f_b, 0x3c_b);
    add_group_one_tests("AND", "Z flag",   and_opcodes, 0x55_b, 0xaa_b, 0x00_b, {}, {Z});
    add_group_one_tests("AND", "N flag",   and_opcodes, 0xd5_b, 0xaa_b, 0x80_b, {}, {N});

    add_group_one_tests("ORA", "no flags", ora_opcodes, 0x54_b, 0x2c_b, 0x7c_b);
    add_group_one_tests("ORA", "Z flag",   ora_opcodes, 0x00_b, 0x00_b, 0x00_b, {}, {Z});
    add_group_one_tests("ORA", "N flag",   ora_opcodes, 0xd4_b, 0x2c_b, 0xfc_b, {}, {N});

    add_group_one_tests("EOR", "no flags", eor_opcodes, 0x6c_b, 0x36_b, 0x5a_b);
    add_group_one_tests("EOR", "Z flag",   eor_opcodes, 0xaa_b, 0xaa_b, 0x00_b, {}, {Z});
    add_group_one_tests("EOR", "N flag",   eor_opcodes, 0x6c_b, 0xb6_b, 0xda_b, {}, {N});

    add_group_one_tests("LDA", "no flags", lda_opcodes, 0xaa_b, 0x55_b, 0x55_b);
    add_group_one_tests("LDA", "Z flag",   lda_opcodes, 0x55_b, 0x00_b, 0x00_b, {}, {Z});
    add_group_one_tests("LDA", "N flag",   lda_opcodes, 0x55_b, 0x80_b, 0x80_b, {}, {N});

    add_group_one_tests("ADC", "no carry",        adc_opcodes, 0x12_b, 0x34_b, 0x46_b);
    add_group_one_tests("ADC", "carry in",        adc_opcodes, 0x12_b, 0x34_b, 0x47_b, {C}, {});
    add_group_one_tests("ADC", "carry out",       adc_opcodes, 0xaa_b, 0x66_b, 0x10_b, {},  {C});
    add_group_one_tests("ADC", "carry in/out",    adc_opcodes, 0xaa_b, 0x55_b, 0x00_b, {C}, {Z, C});
    add_group_one_tests("ADC", "zero, carry out", adc_opcodes, 0xaa_b, 0x56_b, 0x00_b, {},  {Z, C});
    add_group_one_tests("ADC", "positive ovf",    adc_opcodes, 0x75_b, 0x56_b, 0xcb_b, {},  {N, V});
    add_group_one_tests("ADC", "negative ovf",    adc_opcodes, 0x8b_b, 0xaa_b, 0x35_b, {},  {V, C});

    add_group_one_tests("ADC-dec", "no carry",            adc_opcodes, 0x12_b, 0x34_b, 0x46_b, {D},    {D});
    add_group_one_tests("ADC-dec", "bcd carry",           adc_opcodes, 0x19_b, 0x05_b, 0x24_b, {D},    {D});
    add_group_one_tests("ADC-dec", "carry in",            adc_opcodes, 0x12_b, 0x34_b, 0x47_b, {D, C}, {D});
    add_group_one_tests("ADC-dec", "carry in, bcd carry", adc_opcodes, 0x19_b, 0x05_b, 0x25_b, {D, C}, {D});
    add_group_one_tests("ADC-dec", "carry out",           adc_opcodes, 0x84_b, 0x17_b, 0x01_b, {D},    {D, C});
    add_group_one_tests("ADC-dec", "carry in/out",        adc_opcodes, 0x84_b, 0x16_b, 0x01_b, {D, C}, {D, C});
    add_group_one_tests("ADC-dec", "zero",                adc_opcodes, 0x00_b, 0x00_b, 0x00_b, {D},    {D, Z});
    add_group_one_tests("ADC-dec", "zero, carry out",     adc_opcodes, 0x84_b, 0x16_b, 0x00_b, {D},    {D, Z, C});
    add_group_one_tests("ADC-dec", "zero, carry in/out",  adc_opcodes, 0x84_b, 0x15_b, 0x00_b, {D, C}, {D, Z, C});

    add_group_one_tests("SBC", "no borrow",     sbc_opcodes, 0x34_b, 0x12_b, 0x22_b, {C}, {C});
    add_group_one_tests("SBC", "borrow in",     sbc_opcodes, 0x34_b, 0x12_b, 0x21_b, {},  {C});
    add_group_one_tests("SBC", "borrow out",    sbc_opcodes, 0x66_b, 0x67_b, 0xff_b, {C}, {N});
    add_group_one_tests("SBC", "borrow in/out", sbc_opcodes, 0xaa_b, 0xaa_b, 0xff_b, {},  {N});
    add_group_one_tests("SBC", "borrow, zero",  sbc_opcodes, 0xaa_b, 0xa9_b, 0x00_b, {},  {Z, C});
    add_group_one_tests("SBC", "positive ovf",  sbc_opcodes, 0x7f_b, 0xff_b, 0x80_b, {C}, {N, V});
    add_group_one_tests("SBC", "negative ovf",  sbc_opcodes, 0x81_b, 0x01_b, 0x7f_b, {},  {V, C});

    add_group_one_tests("SBC-dec", "no borrow",             sbc_opcodes, 0x34_b, 0x12_b, 0x22_b, {D, C}, {D, C});
    add_group_one_tests("SBC-dec", "bcd borrow",            sbc_opcodes, 0x22_b, 0x16_b, 0x06_b, {D, C}, {D, C});
    add_group_one_tests("SBC-dec", "borrow in",             sbc_opcodes, 0x34_b, 0x12_b, 0x21_b, {D},  {D, C});
    add_group_one_tests("SBC-dec", "borrow in, bcd borrow", sbc_opcodes, 0x22_b, 0x16_b, 0x05_b, {D}, {D, C});
    add_group_one_tests("SBC-dec", "borrow out",            sbc_opcodes, 0x22_b, 0x46_b, 0x76_b, {D, C}, {D});

    add_cond_branch_inst_tests("BCC", 0x90_b, {},  {C});
    add_cond_branch_inst_tests("BCS", 0xb0_b, {C}, {});
    add_cond_branch_inst_tests("BNE", 0xd0_b, {},  {Z});
    add_cond_branch_inst_tests("BEQ", 0xf0_b, {Z}, {});
    add_cond_branch_inst_tests("BPL", 0x10_b, {},  {N});
    add_cond_branch_inst_tests("BMI", 0x30_b, {N}, {});
    add_cond_branch_inst_tests("BVC", 0x50_b, {},  {V});
    add_cond_branch_inst_tests("BVS", 0x70_b, {V}, {});

    add_shift_inst_test("ASL", "no flags", asl_opcodes, 0x35_b, 0x6a_b);
    add_shift_inst_test("ASL", "N flag", asl_opcodes, 0x55_b, 0xaa_b, {}, {N});
    add_shift_inst_test("ASL", "C flag", asl_opcodes, 0xaa_b, 0x54_b, {}, {C});
    add_shift_inst_test("ASL", "N,C flags", asl_opcodes, 0xca_b, 0x94_b, {}, {N, C});
    add_shift_inst_test("ASL", "Z flag", asl_opcodes, 0x00_b, 0x00_b, {}, {Z});
    add_shift_inst_test("ASL", "Z,C flags", asl_opcodes, 0x80_b, 0x00_b, {}, {Z,C});

    add_shift_inst_test("ASL", "no flags, ignore carry", asl_opcodes, 0x35_b, 0x6a_b, {C}, {});
    add_shift_inst_test("ASL", "N flag, ignore carry", asl_opcodes, 0x55_b, 0xaa_b, {C}, {N});
    add_shift_inst_test("ASL", "C flag, ignore carry", asl_opcodes, 0xaa_b, 0x54_b, {C}, {C});
    add_shift_inst_test("ASL", "N,C flags, ignore carry", asl_opcodes, 0xca_b, 0x94_b, {C}, {N, C});
    add_shift_inst_test("ASL", "Z flag, ignore carry", asl_opcodes, 0x00_b, 0x00_b, {C}, {Z});
    add_shift_inst_test("ASL", "Z,C flags, ignore carry", asl_opcodes, 0x80_b, 0x00_b, {C}, {Z,C});

    add_shift_inst_test("LSR", "no flags", lsr_opcodes, 0x6a_b, 0x35_b);
    add_shift_inst_test("LSR", "C flag", lsr_opcodes, 0x55_b, 0x2a_b, {}, {C});
    add_shift_inst_test("LSR", "Z flag", lsr_opcodes, 0x00_b, 0x00_b, {}, {Z});
    add_shift_inst_test("LSR", "Z,C flags", lsr_opcodes, 0x01_b, 0x00_b, {}, {Z,C});

    add_shift_inst_test("LSR", "no flags, igore carry", lsr_opcodes, 0x6a_b, 0x35_b, {C}, {});
    add_shift_inst_test("LSR", "C flag, igore carry", lsr_opcodes, 0x55_b, 0x2a_b, {C}, {C});
    add_shift_inst_test("LSR", "Z flag, igore carry", lsr_opcodes, 0x00_b, 0x00_b, {C}, {Z});
    add_shift_inst_test("LSR", "Z,C flags, igore carry", lsr_opcodes, 0x01_b, 0x00_b, {C}, {Z,C});

    add_shift_inst_test("ROL", "no flags", rol_opcodes, 0x35_b, 0x6a_b);
    add_shift_inst_test("ROL", "N flag", rol_opcodes, 0x55_b, 0xaa_b, {}, {N});
    add_shift_inst_test("ROL", "C flag", rol_opcodes, 0xaa_b, 0x54_b, {}, {C});
    add_shift_inst_test("ROL", "N,C flags", rol_opcodes, 0xca_b, 0x94_b, {}, {N, C});
    add_shift_inst_test("ROL", "Z flag", rol_opcodes, 0x00_b, 0x00_b, {}, {Z});
    add_shift_inst_test("ROL", "Z,C flags", rol_opcodes, 0x80_b, 0x00_b, {}, {Z,C});

    add_shift_inst_test("ROL", "no flags, carry in", rol_opcodes, 0x35_b, 0x6b_b, {C}, {});
    add_shift_inst_test("ROL", "N flag, carry in", rol_opcodes, 0x55_b, 0xab_b, {C}, {N});
    add_shift_inst_test("ROL", "C flag, carry in", rol_opcodes, 0xaa_b, 0x55_b, {C}, {C});
    add_shift_inst_test("ROL", "N,C flags, carry in", rol_opcodes, 0xca_b, 0x95_b, {C}, {N, C});

    add_shift_inst_test("ROR", "no flags", ror_opcodes, 0x6a_b, 0x35_b);
    add_shift_inst_test("ROR", "C flag", ror_opcodes, 0x55_b, 0x2a_b, {}, {C});
    add_shift_inst_test("ROR", "Z flag", ror_opcodes, 0x00_b, 0x00_b, {}, {Z});
    add_shift_inst_test("ROR", "Z,C flags", ror_opcodes, 0x01_b, 0x00_b, {}, {Z,C});

    add_shift_inst_test("ROR", "N flag, carry in", ror_opcodes, 0xaa_b, 0xd5_b, {C}, {N});
    add_shift_inst_test("ROR", "N,C flag, carry in", ror_opcodes, 0x55_b, 0xaa_b, {C}, {N, C});

    #undef Z
    #undef N
    #undef C
    #undef V
    #undef D
    // clanf-format on

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
