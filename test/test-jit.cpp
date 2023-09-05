#include "../src/jit.h"
#include "../src/codegen.h"
#include "../src/control_flow.h"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>

using namespace emu::literals;

namespace {
llvm::ExitOnError exit_on_error;
}

struct ParamType {
    std::string name;
    std::vector<emu::byte_t> instructions;
    uint64_t expected_cycles;
    emu::word_t return_address{1234_w};
    emu::CPU initial_cpu_state{};
    bool do_rti{};
    emu::byte_t expected_sr{};
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
        std::ranges::copy(params.instructions,
                          _bus.memory_space.begin() + static_cast<size_t>(_cpu.PC));

        // Add return instruction
        emu::byte_t const ret_instruction = params.do_rti ? 0x40_b : 0x60_b;
        _bus.memory_space[static_cast<size_t>(_cpu.PC) + params.instructions.size()] =
            ret_instruction;

        // Set up stack
        _bus.write(0x1ff_w, emu::get_hi(params.return_address));
        _bus.write(0x1fe_w, emu::get_lo(params.return_address));

        // Add SR to the stack if the return instruction is an RTI
        if (params.do_rti) {
            _bus.write(0x1fd_w, params.expected_sr);
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
    auto const control_flow = emu::build_control_flow(_bus, params.initial_cpu_state.PC);
    auto module = emu::codegen(_context, control_flow);
    auto err = emu::materialize(*_jit, _context, std::move(module));
    ASSERT_FALSE(test_error(std::move(err)));

    auto fn_ex = _jit->lookup(get_fn_name());
    ASSERT_TRUE(test_expected(fn_ex));
    auto* const fn = fn_ex.get().toPtr<emu::jit_fn_t>();

    auto const cycles = fn(_cpu, _bus);

    EXPECT_EQ(cycles, params.expected_cycles + 6);

    EXPECT_EQ(_cpu.PC, params.return_address);
    EXPECT_EQ(_cpu.SP, 0xff_b);
    EXPECT_EQ(_cpu.SR.get(), params.expected_sr | 0x30_b);
}

// std::string_view name;
// std::vector<emu::byte_t> instructions;
// uint64_t expected_cycles;
// emu::word_t return_address{1234_w};
// emu::CPU initial_cpu_state{};
// bool do_rti{};
// emu::byte_t expected_sr{};

std::vector<ParamType> test_cases{
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
        .expected_sr = 0xff_b,
    },
};

INSTANTIATE_TEST_SUITE_P(Test, TestJitCodegen, ::testing::ValuesIn(test_cases));
