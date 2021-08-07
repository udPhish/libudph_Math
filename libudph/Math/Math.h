#pragma once
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace UD
{
namespace Math
{
namespace Concepts
{
template<class T>
concept HasAddAssign = requires(T t1, T t2)
{
  {
    t1 += t2
    } -> std::convertible_to<T&>;
};
template<class T>
concept HasAdd = requires(T t1, T t2)
{
  {
    t1 + t2
    } -> std::convertible_to<T>;
};
template<class T>
concept HasSubtractAssign = requires(T t1, T t2)
{
  {
    t1 -= t2
    } -> std::convertible_to<T&>;
};
template<class T>
concept HasSubtract = requires(T t1, T t2)
{
  {
    t1 - t2
    } -> std::convertible_to<T>;
};
template<class T>
concept HasMultiplyAssign = requires(T t1, T t2)
{
  {
    t1 *= t2
    } -> std::convertible_to<T&>;
};
template<class T>
concept HasMultiply = requires(T t1, T t2)
{
  {
    t1* t2
    } -> std::convertible_to<T>;
};
template<class T>
concept HasDivideAssign = requires(T t1, T t2)
{
  {
    t1 /= t2
    } -> std::convertible_to<T&>;
};
template<class T>
concept HasDivide = requires(T t1, T t2)
{
  {
    t1 / t2
    } -> std::convertible_to<T>;
};
template<class T>
concept Addable = HasAdd<T> && HasAddAssign<T>;
template<class T>
concept Subtractable = HasSubtract<T> && HasSubtractAssign<T>;
template<class T>
concept Multipliable = HasMultiply<T> && HasMultiplyAssign<T>;
template<class T>
concept Dividable = HasDivide<T> && HasDivideAssign<T>;
template<class T>
concept Incrementable = requires(T t)
{
  {
    t++
    } -> std::convertible_to<T>;
  {
    ++t
    } -> std::convertible_to<T&>;
};
template<class T>
concept Decrementable = requires(T t)
{
  {
    t--
    } -> std::convertible_to<T>;
  {
    --t
    } -> std::convertible_to<T&>;
};

template<class T>
concept Arithmeticable
    = Addable<T> && Subtractable<T> && Multipliable<T> && Dividable<T>;
}  // namespace Concepts
namespace Type
{
using Short    = std::int_least16_t;
using UShort   = std::make_unsigned_t<Short>;
using Long     = std::intmax_t;
using ULong    = std::make_unsigned_t<Long>;
using Integer  = std::int_fast32_t;
using UInteger = std::make_unsigned_t<Integer>;
using Double   = double;
using Float    = float;
}  // namespace Type
template<class T, class U>
constexpr U Clamp(const T& size)
{
  static_assert(
      std::numeric_limits<T>::is_integer && std::numeric_limits<U>::is_integer,
      "Function requires integer types.");
  if (std::numeric_limits<T>::is_signed == std::numeric_limits<U>::is_signed)
  {
    if (sizeof(T) <= sizeof(U))
      return static_cast<U>(size);
    if (sizeof(T) > sizeof(U))
    {
      if (size > 0)
        return static_cast<U>(
            std::min(static_cast<T>(std::numeric_limits<U>::max()), size));
      else
        return static_cast<U>(
            std::max(static_cast<T>(std::numeric_limits<U>::min()), size));
    }
  }
  else
  {
    if (std::numeric_limits<T>::is_signed)
    {
      if (size < 0)
        return 0;
      if (sizeof(T) - 1 <= sizeof(U))
        return static_cast<U>(size);
      if (sizeof(T) - 1 > sizeof(U))
        return static_cast<U>(
            std::min(static_cast<T>(std::numeric_limits<U>::max()), size));
    }
    else
    {
      if (sizeof(T) <= sizeof(U) - 1)
        return static_cast<U>(size);
      if (sizeof(T) > sizeof(U) - 1)
        return static_cast<U>(
            std::min(static_cast<T>(std::numeric_limits<U>::max()), size));
    }
  }
}
}  // namespace Math
}  // namespace UD
