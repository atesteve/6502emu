#pragma once

#include "cpu.h"
#include "jit.h"

#include <llvm/Support/ThreadPool.h>

#include <string_view>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace emu {

struct JitFn;

class Emulator {
public:
    uint64_t run();

    auto const& get_cpu() const noexcept { return _cpu; }
    auto const& get_bus() const noexcept { return _bus; }
    void initialize(std::string_view image_file, word_t load_address, word_t entry_point);
    uint64_t call_function(word_t addr);

    Emulator();
    ~Emulator();

private:
    void jit_function(word_t addr);
    JitFn* get_jit_fn(word_t addr);

    CPU _cpu{};
    Bus _bus{};
    uint64_t _intr_clock_counter{};
    uint64_t _jit_clock_counter{};
    std::mutex _map_mutex;
    std::vector<std::unique_ptr<JitFn>> _jit_functions;
    std::vector<jit_fn_t> _jit_functions_cache;
    std::unique_ptr<llvm::ThreadPool> _thread_pool{};
};

} // namespace emu
