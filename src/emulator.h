#pragma once

#include "cpu.h"

#include <string_view>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace emu {

struct JitFn;

class Emulator {
public:
    uint64_t run();

    auto get_clock_counter() const noexcept { return _clock_counter; }
    auto const& get_cpu() const noexcept { return _cpu; }
    auto const& get_bus() const noexcept { return _bus; }
    void initialize(std::string_view image_file, word_t load_address, word_t entry_point);
    uint64_t call_function(word_t addr);

    Emulator();
    ~Emulator();

private:
    JitFn* get_jit_fn(word_t addr);

    CPU _cpu{};
    Bus _bus{};
    uint64_t _clock_counter{};
    std::mutex _map_mutex;
    std::unordered_map<word_t, std::unique_ptr<JitFn>> _jit_functions;
};

} // namespace emu
