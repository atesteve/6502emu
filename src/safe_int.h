#pragma once

#include <fmt/core.h>

#include <compare>
#include <concepts>
#include <functional>
#include <type_traits>

#define ALWAYS_INLINE __attribute__((always_inline))

namespace emu {
namespace detail {
enum class default_token {};
}

template<std::integral Int, typename Token = detail::default_token>
class SafeInt {
    template<std::integral Int2, typename Token2>
    friend class SafeInt;

public:
    ALWAYS_INLINE constexpr SafeInt() noexcept = default;
    ALWAYS_INLINE explicit constexpr SafeInt(Int v) noexcept
        : _v{v}
    {}

    template<std::integral Int2>
        requires(!std::is_same_v<Int, Int2>)
    ALWAYS_INLINE explicit constexpr SafeInt(Int2 v) noexcept
        : _v{static_cast<Int>(v)}
    {}

    ALWAYS_INLINE constexpr SafeInt(SafeInt const& o) noexcept
        : _v{o._v}
    {}

    ALWAYS_INLINE constexpr SafeInt& operator=(SafeInt const& o) noexcept
    {
        _v = o._v;
        return *this;
    }

    template<std::integral Int2, typename Token2>
        requires(!std::is_same_v<Int, Int2>)
    ALWAYS_INLINE explicit constexpr operator SafeInt<Int2, Token2>() const noexcept
    {
        return SafeInt<Int2, Token2>{static_cast<Int2>(_v)};
    }

    template<std::integral Int2>
    ALWAYS_INLINE explicit constexpr operator Int2() const noexcept
    {
        return static_cast<Int2>(_v);
    }

    ALWAYS_INLINE constexpr auto operator<=>(SafeInt const&) const noexcept = default;

    ALWAYS_INLINE constexpr SafeInt operator+(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v + o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator-(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v - o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator*(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v * o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator/(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v / o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator%(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v % o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator&(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v & o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator|(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v | o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator^(SafeInt const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v ^ o._v)};
    }

    ALWAYS_INLINE constexpr SafeInt& operator+=(SafeInt const& o) noexcept
    {
        _v += o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator-=(SafeInt const& o) noexcept
    {
        _v -= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator*=(SafeInt const& o) noexcept
    {
        _v *= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator/=(SafeInt const& o) noexcept
    {
        _v /= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator%=(SafeInt const& o) noexcept
    {
        _v %= o._v;
        return *this;
    }


    ALWAYS_INLINE constexpr SafeInt& operator&=(SafeInt const& o) noexcept
    {
        _v &= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator|=(SafeInt const& o) noexcept
    {
        _v |= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator^=(SafeInt const& o) noexcept
    {
        _v ^= o._v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt operator~() const noexcept
    {
        return SafeInt{static_cast<Int>(~_v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator-() const noexcept
    {
        return SafeInt{static_cast<Int>(-_v)};
    }

    ALWAYS_INLINE constexpr SafeInt operator+() const noexcept { return _v; }

    ALWAYS_INLINE constexpr SafeInt& operator++() noexcept
    {
        ++_v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt operator++(int) noexcept
    {
        auto copy{*this};
        ++_v;
        return copy;
    }

    ALWAYS_INLINE constexpr SafeInt& operator--() noexcept
    {
        --_v;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt operator--(int) noexcept
    {
        auto copy{*this};
        --_v;
        return copy;
    }

    ALWAYS_INLINE constexpr SafeInt operator>>(int bits) const noexcept
    {
        return SafeInt{static_cast<Int>(_v >> bits)};
    }

    ALWAYS_INLINE constexpr SafeInt operator<<(int bits) const noexcept
    {
        return SafeInt{static_cast<Int>(_v << bits)};
    }

    ALWAYS_INLINE constexpr SafeInt& operator>>=(int bits) noexcept
    {
        _v >>= bits;
        return *this;
    }

    ALWAYS_INLINE constexpr SafeInt& operator<<=(int bits) noexcept
    {
        _v <<= bits;
        return *this;
    }

    template<std::integral Int2, typename Token2>
        requires(sizeof(Int2) < sizeof(Int))
    ALWAYS_INLINE constexpr SafeInt operator+(SafeInt<Int2, Token2> const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v + o._v)};
    }

    template<std::integral Int2, typename Token2>
        requires(sizeof(Int2) < sizeof(Int))
    ALWAYS_INLINE constexpr SafeInt operator-(SafeInt<Int2, Token2> const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v - o._v)};
    }

    template<std::integral Int2, typename Token2>
        requires(sizeof(Int2) < sizeof(Int))
    ALWAYS_INLINE constexpr SafeInt operator*(SafeInt<Int2, Token2> const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v * o._v)};
    }

    template<std::integral Int2, typename Token2>
        requires(sizeof(Int2) < sizeof(Int))
    ALWAYS_INLINE constexpr SafeInt operator/(SafeInt<Int2, Token2> const& o) const noexcept
    {
        return SafeInt{static_cast<Int>(_v / o._v)};
    }

private:
    Int _v{};
};

} // namespace emu

template<std::integral Int, typename Token>
struct std::hash<emu::SafeInt<Int, Token>> : std::hash<Int> {
    constexpr std::size_t operator()(emu::SafeInt<Int, Token> n) const
    {
        return std::hash<Int>::operator()(static_cast<Int>(n));
    }
};

template<std::integral Int, typename Token>
struct fmt::formatter<emu::SafeInt<Int, Token>> : public fmt::formatter<Int> {
    constexpr auto format(emu::SafeInt<Int, Token> const& n, fmt::format_context& ctx)
    {
        return fmt::formatter<Int>::format(static_cast<Int>(n), ctx);
    }
};
