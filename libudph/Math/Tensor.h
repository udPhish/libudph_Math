#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <ranges>
#include <tuple>

#include <libudph/Class/Event.h>
#include <libudph/Class/Interface.h>
#include <libudph/Class/Pack.h>
#include <libudph/Class/SFINAE.h>
#include <libudph/Class/Traits.h>
#include <libudph/Container/Container.h>
#include <libudph/Math/udMath.h>

namespace UD
{
template<Math::Type::ULong S, Math::Type::ULong... Ss>
struct For
{
  template<Math::Type::ULong, Math::Type::ULong...>
  friend struct For;

 private:
  template<class T, class... Ts, std::convertible_to<Math::Type::ULong>... Is>
    requires Concept::
        Invocable<T, Ts..., decltype(Ss)..., Is..., Math::Type::ULong> &&(
            sizeof...(Ss)
            > 0) static void CallHelper(T&& t, Ts&&... ts, Is&&... is)
    {
      for (Math::Type::ULong i = 0; i < S; ++i)
      {
        For<Ss...>::template CallHelper<T, Ts..., Is...>(
            std::forward<T>(t),
            std::forward<Ts>(ts)...,
            std::forward<Is>(is)...,
            i);
      }
    }
    template<class T, class... Ts, std::convertible_to<Math::Type::ULong>... Is>
      requires Concept::Invocable<T, Ts..., Is..., Math::Type::ULong> &&(
          sizeof...(Ss)
          == 0) static void CallHelper(T&& t, Ts&&... ts, Is&&... is)
      {
        for (Math::Type::ULong i = 0; i < S; ++i)
        {
          t(std::forward<Ts>(ts)..., std::forward<Is>(is)..., i);
        }
      }
      template<class T>
      struct CallReverseHelper;

      template<Math::Type::ULong... Vs>
      struct CallReverseHelper<Pack::ValuesPack<Vs...>>
      {
        template<class T, class... Ts>
        static void Call(T&& t, Ts&&... ts)
        {
          For<Vs...>::template CallHelper<T, Ts...>(std::forward<T>(t),
                                                    std::forward<Ts>(ts)...);
        }
      };

     public:
      template<class T, class... Ts>
      static inline void Call(T&& t, Ts&&... ts)
      {
        CallHelper<T, Ts...>(std::forward<T>(t), std::forward<Ts>(ts)...);
      }
      template<class T, class... Ts>
      static inline void CallReverse(T&& t, Ts&&... ts)
      {
        CallReverseHelper<typename Pack::ValuesPack<S, Ss...>::Reverse>::Call(
            std::forward<T>(t),
            std::forward<Ts>(ts)...);
      }
};

}  // namespace UD
namespace UD::Concepts
{
template<class From, class To>
concept ConvertibleTo = std::convertible_to<From, To>;
template<class From, class To>
concept NotConvertibleTo = !ConvertibleTo<From, To>;
template<class From, class To>
concept ExplicitlyConvertibleTo = requires(From from)
{
  {
    static_cast<To>(from)
    } -> std::same_as<To>;
};
template<class From, class To>
concept OnlyExplicitlyConvertibleTo
    = ExplicitlyConvertibleTo<From, To> && NotConvertibleTo<From, To>;
template<class From, class To>
concept NotOnlyExplicitlyConvertibleTo = !OnlyExplicitlyConvertibleTo<From, To>;
template<class T>
concept EqualityComparable = std::equality_comparable<T>;
template<class T, class U>
concept EqualityComparableWith = std::equality_comparable_with<T, U>;
template<class T>
concept TotallyOrdered = std::totally_ordered<T>;
template<class T, class U>
concept TotallyOrderedWith = std::totally_ordered_with<T, U>;
template<class T, class U>
concept IteratorTo = std::input_or_output_iterator<T> && requires(T t)
{
  {
    *t
    } -> std::same_as<std::remove_reference<T>&>;
};
template<class T, class U>
concept BinaryOperation = requires(T t, U u, U v)
{
  {
    t(u, v)
    } -> std::convertible_to<typename std::remove_cvref<U>::type>;
};
}  // namespace UD::Concepts
namespace UD::Math
{
template<std::ranges::range T, std::ranges::range U>
  requires std::same_as < typename std::remove_cvref<T>::type,
typename std::remove_cvref<U>::type > auto& zip_transform(
    T&  lhs,
    U&& rhs,
    UD::Concepts::BinaryOperation<decltype(*std::ranges::begin(lhs))> auto&& op)
{
  std::ranges::transform(lhs,
                         std::forward<U>(rhs),
                         std::ranges::begin(lhs),
                         std::forward<decltype(op)>(op));
  return lhs;
}
template<std::ranges::range T, std::ranges::range U, std::ranges::range V>
  requires std::same_as < typename std::remove_cvref<T>::type,
typename std::remove_cvref<U>::type
    > &&std::same_as<typename std::remove_cvref<U>::type,
                     typename std::remove_cvref<V>::type> auto&
      zip_transform(T&& lhs,
                    U&& rhs,
                    V&  ret,
                    UD::Concepts::BinaryOperation<
                        decltype(*std::ranges::begin(lhs))> auto&& op)
{
  std::ranges::transform(std::forward<T>(lhs),
                         std::forward<U>(rhs),
                         std::ranges::begin(ret),
                         std::forward<decltype(op)>(op));
  return ret;
}
}  // namespace UD::Math
namespace UD::Concepts::Tensor
{
template<class T>
concept Arithmeticable = requires(T t, const T& ct, const T& cu)
{
  {ct + cu};
  {t += cu};
  {ct - cu};
  {t -= cu};
  {ct / cu};
  {t /= cu};
  {ct * cu};
  {t *= cu};
};
template<class T>
concept Range = std::ranges::range<T>;
template<class T>
concept NotRange = !Range<T>;
template<class T>
concept ArithmeticableRange
    = Range<T> && Arithmeticable<std::ranges::range_value_t<T>>;
template<class T>
concept CompoundArithmeticableRange = ArithmeticableRange<
    T> && ArithmeticableRange<std::ranges::range_value_t<T>>;
template<class T>
concept Tensor = Arithmeticable<T> && ArithmeticableRange<T>;
template<class T>
concept NotTensor = !Tensor<T>;
template<class T>
concept Vector = Tensor<T>;
template<class T>
concept Matrix = Tensor<T> && Vector<std::ranges::range_value_t<T>>;
}  // namespace UD::Concepts::Tensor
namespace UD::Tensor
{
template<UD::Concepts::Tensor::Arithmeticable _T, UD::Math::Type::ULong _Size>
struct TensorRaw;
}
namespace UD::Traits::Tensor
{
template<class T>
struct Size;

template<class T, Math::Type::ULong _Size>
struct Size<UD::Tensor::TensorRaw<T, _Size>>
{
  static const Math::Type::ULong value = _Size;
};
template<class T, Math::Type::ULong _Size>
struct Size<std::array<T, _Size>>
{
  static const Math::Type::ULong value = _Size;
};
}  // namespace UD::Traits::Tensor
namespace UD::Concepts::Tensor
{
template<class T>
concept SizedRange = Range<T> && std::ranges::sized_range<T>;
template<class T>
concept CSizedRange = SizedRange<T> && requires(T t)
{
  {UD::Traits::Tensor::Size<T>::value};
};
}  // namespace UD::Concepts::Tensor
namespace std
{
template<class T>
  requires UD::Concepts::Tensor::CSizedRange<T>
constexpr auto size(const T& c)
{
  return UD::Traits::Tensor::Size<T>::value;
}
template<class T>
  requires UD::Concepts::Tensor::CSizedRange<T>
struct tuple_size<T>
{
  static const size_t value = UD::Traits::Tensor::Size<T>::value;
};
}  // namespace std
namespace UD::Traits::Tensor
{
template<class T>
struct SatisfiesDefault : public std::true_type
{
};
template<class T>
struct SatisfiesTensor : public std::false_type
{
};
template<UD::Concepts::Tensor::Tensor T>
struct SatisfiesTensor<T> : public std::true_type
{
};
template<Concepts::Tensor::Range T>
struct Element
{
  using type = std::ranges::range_value_t<T>;
};
template<Concepts::Tensor::Range T,
         template<class> class Satisfies = SatisfiesDefault>
struct BaseElement
{
  using type = Element<T>::type;
};
template<Concepts::Tensor::Range T, template<class> class Satisfies>
requires Concepts::Tensor::Range<typename Element<T>::type> && Satisfies<
    typename Element<T>::type>::value struct BaseElement<T, Satisfies>
{
  using type = typename BaseElement<typename Element<T>::type>::type;
};

template<Concepts::Tensor::Range T,
         class Element,
         template<class> class Satisfies = SatisfiesDefault>
struct IsRangeOf : public std::false_type
{
};
template<Concepts::Tensor::Range T,
         class _Element,
         template<class>
         class Satisfies>
  requires(
      Satisfies<T>::value&& std::same_as<typename Element<T>::type, _Element>)
|| IsRangeOf<typename Element<T>::type, _Element, Satisfies>::
        value struct IsRangeOf<T, _Element, Satisfies> : public std::true_type
{
};
template<class T, class Element>
struct IsTensorOf : public IsRangeOf<T, Element, SatisfiesTensor>
{
};
}  // namespace UD::Traits::Tensor
namespace UD::Concepts::Tensor
{
template<class T,
         class Element,
         template<class> class Satisfies = Traits::Tensor::SatisfiesDefault>
concept RangeOf
    = Range<T> && Traits::Tensor::IsRangeOf<T, Element, Satisfies>::value;
template<class T,
         class Element,
         template<class> class Satisfies = Traits::Tensor::SatisfiesDefault>
concept SimpleRangeOf = RangeOf<T, Element, Satisfies> && std::
    same_as<typename Traits::Tensor::Element<T>::type, Element>;
template<class T,
         template<class> class Satisfies = Traits::Tensor::SatisfiesDefault>
concept BaseRange
    = SimpleRangeOf<T,
                    typename Traits::Tensor::BaseElement<T, Satisfies>::type,
                    Satisfies>;
template<class T,
         class Element,
         template<class> class Satisfies = Traits::Tensor::SatisfiesDefault>
concept BaseRangeOf
    = BaseRange<T, Satisfies> && SimpleRangeOf<T, Element, Satisfies>;

template<class T, class Element>
concept TensorOf
    = Tensor<T> && RangeOf<T, Element, Traits::Tensor::SatisfiesTensor>;
template<class T, class Element>
concept NotTensorOf = !TensorOf<T, Element>;
template<class T, class Element>
concept SimpleTensorOf
    = TensorOf<T, Element> && SimpleRangeOf<T,
                                            Element,
                                            Traits::Tensor::SatisfiesTensor>;
template<class T>
concept BaseTensor = Tensor<
    T> && SimpleTensorOf<T, typename Traits::Tensor::BaseElement<T>::type>;
template<class T>
concept NotBaseTensor = !BaseTensor<T>;
template<class T, class Element>
concept BaseTensorOf = BaseTensor<T> && SimpleTensorOf<T, Element>;

template<class T, class Element>
concept VectorOf = Vector<T> && SimpleTensorOf<T, Element>;
template<class T>
concept BaseVector
    = VectorOf<T, typename Traits::Tensor::Element<T>::type> && BaseTensorOf<
        T,
        typename Traits::Tensor::Element<T>::type>;
template<class T>
concept NotBaseVector = !BaseVector<T>;
template<class T, class Element>
concept BaseVectorOf = BaseVector<T> && VectorOf<T, Element>;

template<class T, class Element>
concept MatrixOf = Matrix<T> && TensorOf<T, Element> && VectorOf<
    typename Traits::Tensor::Element<T>::type,
    Element>;
template<class T>
concept BaseMatrix
    = Matrix<T> && BaseVector<typename Traits::Tensor::Element<T>::type>;
template<class T, class Element>
concept BaseMatrixOf
    = BaseMatrix<T> && BaseVectorOf<typename Traits::Tensor::Element<T>::type,
                                    Element>;
}  // namespace UD::Concepts::Tensor
namespace UD::Traits::Tensor
{
template<class T, class... Element>
struct Rank
{
  static consteval UD::Math::Type::ULong test()
  {
    return 0;
  }
  static constexpr UD::Math::Type::ULong value = test();
};
template<class T, class _Element>
  requires Concepts::Tensor::VectorOf<T, _Element>
struct Rank<T, _Element>
{
  static consteval UD::Math::Type::ULong test()
  {
    return 1;
  }
  static constexpr UD::Math::Type::ULong value = test();
};
template<class T, class _Element>
  requires Concepts::Tensor::TensorOf<T, _Element>
struct Rank<T, _Element>
{
  static consteval UD::Math::Type::ULong test()
  {
    return 1 + Rank<typename Element<T>::type, _Element>::test();
  }
  static constexpr UD::Math::Type::ULong value = test();
};
template<Concepts::Tensor::Tensor T>
struct Rank<T>
{
  static consteval UD::Math::Type::ULong test()
  {
    return Rank<T, typename BaseElement<T, SatisfiesTensor>::type>::test();
  }
  static constexpr UD::Math::Type::ULong value = test();
};

namespace detail
{
template<class T, UD::Math::Type::ULong R>
concept ValidRank_ElementAtRank = R <= Rank<T>::value&& R > 0;
template<class T, UD::Math::Type::ULong D>
concept ValidDepth_ElementAtDepth = D < Rank<T>::value;
}  // namespace detail
template<Concepts::Tensor::Range T, Math::Type::ULong R>
  requires detail::ValidRank_ElementAtRank<T, R>
struct ElementAtRank : public Element<T>
{
};

template<Concepts::Tensor::Range T, Math::Type::ULong R>
  requires detail::ValidRank_ElementAtRank<T, R> &&(
      Rank<T>::value - R > 0) struct ElementAtRank<T, R>
      : public ElementAtRank<typename Element<T>::type, R>
  {
  };
  template<Concepts::Tensor::Range T, Math::Type::ULong R>
    requires detail::ValidDepth_ElementAtDepth<T, R>
  struct ElementAtDepth : public Element<T>
  {
  };

  template<Concepts::Tensor::Range T, Math::Type::ULong R>
    requires detail::ValidDepth_ElementAtDepth<T, R> &&(
        R > 0) struct ElementAtDepth<T, R>
        : public ElementAtDepth<typename Element<T>::type, R - 1>
    {
    };
    namespace detail
    {
    template<class T>
    struct SizePackHelper;
    template<class T>
    struct SizePackHelper
    {
      using type = Pack::ValuesPack<Size<T>::value>;
    };
    template<class T>
      requires Concepts::Tensor::Tensor<typename Element<T>::type>
    struct SizePackHelper<T>
    {
      using type = typename Pack::ValuesPackConcat<
          Pack::ValuesPack<Size<T>::value>,
          typename SizePackHelper<typename Element<T>::type>::type>::type;
    };
    }  // namespace detail
    template<class T>
    using SizePack = typename detail::SizePackHelper<T>::type;
}  // namespace UD::Traits::Tensor
namespace UD::Concepts::Tensor
{
namespace detail
{
template<class T>
struct IsSquare : public std::false_type
{
};
template<BaseTensor T>
struct IsSquare<T> : public std::true_type
{
};
template<Tensor T>
  requires CSizedRange<T> &&(
      UD::Traits::Tensor::Size<T>::value
      == UD::Traits::Tensor::Size<
          typename UD::Traits::Tensor::Element<T>::type>::value)
      && IsSquare<typename UD::Traits::Tensor::Element<T>::type>::value
      struct IsSquare<T> : public std::true_type
  {
  };
}  // namespace detail
template<class T>
concept Square = Tensor<T> && CSizedRange<T> && detail::IsSquare<T>::value;
namespace detail
{

  template<class T, class U>
  struct IsGreaterRelative : public std::false_type
  {
  };
  template<Concepts::Tensor::Tensor T, Concepts::Tensor::Tensor U>
    requires NotBaseTensor<T> && NotBaseTensor<U> &&(
        Traits::Tensor::Rank<T>::value == Traits::Tensor::Rank<U>::value)
        && (Traits::Tensor::Size<T>::value
            > Traits::Tensor::Size<U>::value) struct IsGreaterRelative<T, U>
        : public IsGreaterRelative<typename Traits::Tensor::Element<T>::type,
                                   typename Traits::Tensor::Element<U>::type>
    {
    };
    template<Concepts::Tensor::BaseTensor T, Concepts::Tensor::BaseTensor U>
      requires(Traits::Tensor::Rank<T>::value == Traits::Tensor::Rank<U>::value)
    &&(Traits::Tensor::Size<T>::value
       > Traits::Tensor::Size<U>::value) struct IsGreaterRelative<T, U>
        : public std::true_type
    {
    };
    template<class T, class U>
    struct IsLesserRelative : public std::false_type
    {
    };
    template<Concepts::Tensor::Tensor T, Concepts::Tensor::Tensor U>
      requires NotBaseTensor<T> && NotBaseTensor<U> &&(
          Traits::Tensor::Rank<T>::value == Traits::Tensor::Rank<U>::value)
          && (Traits::Tensor::Size<T>::value
              < Traits::Tensor::Size<U>::value) struct IsLesserRelative<T, U>
          : public IsLesserRelative<typename Traits::Tensor::Element<T>::type,
                                    typename Traits::Tensor::Element<U>::type>
      {
      };
      template<Concepts::Tensor::BaseTensor T, Concepts::Tensor::BaseTensor U>
        requires(Traits::Tensor::Rank<T>::value
                 == Traits::Tensor::Rank<U>::value)
      &&(Traits::Tensor::Size<T>::value
         < Traits::Tensor::Size<U>::value) struct IsLesserRelative<T, U>
          : public std::true_type
      {
      };

      template<class T, class U>
      struct IsIdenticalRelative : public std::false_type
      {
      };
      template<Concepts::Tensor::Tensor T, Concepts::Tensor::Tensor U>
        requires NotBaseTensor<T> && NotBaseTensor<U> &&(
            Traits::Tensor::Rank<T>::value == Traits::Tensor::Rank<U>::value)
            && (Traits::Tensor::Size<T>::value
                == Traits::Tensor::Size<U>::value) struct IsIdenticalRelative<T,
                                                                              U>
            : public IsIdenticalRelative<
                  typename Traits::Tensor::Element<T>::type,
                  typename Traits::Tensor::Element<U>::type>
        {
        };
        template<Concepts::Tensor::BaseTensor T, Concepts::Tensor::BaseTensor U>
          requires(Traits::Tensor::Rank<T>::value
                   == Traits::Tensor::Rank<U>::value)
        &&(Traits::Tensor::Size<T>::value
           == Traits::Tensor::Size<U>::value) struct IsIdenticalRelative<T, U>
            : public std::true_type
        {
        };
        template<class T, class U, Math::Type::Long... Sizes>
        struct IsRelative : public std::false_type
        {
        };

        template<Concepts::Tensor::Tensor T,
                 Concepts::Tensor::Tensor U,
                 Math::Type::Long         Size,
                 Math::Type::Long... Sizes>
          requires(Traits::Tensor::Rank<T>::value
                   == Traits::Tensor::Rank<U>::value)
        &&(Traits::Tensor::Size<T>::value
           == Traits::Tensor::Size<U>::value
                  + Size) struct IsRelative<T, U, Size, Sizes...>
            : public IsRelative<typename Traits::Tensor::Element<T>::type,
                                typename Traits::Tensor::Element<U>::type,
                                Sizes...>
        {
        };
        template<Concepts::Tensor::Tensor T,
                 Concepts::Tensor::Tensor U,
                 Math::Type::Long         Size>
          requires(Traits::Tensor::Rank<T>::value
                   == Traits::Tensor::Rank<U>::value)
        &&(Traits::Tensor::Size<T>::value
           == Traits::Tensor::Size<U>::value
                  + Size) struct IsRelative<T, U, Size> : public std::true_type
        {
        };
        template<Concepts::Tensor::Tensor T, Concepts::Tensor::Tensor U>
          requires(Traits::Tensor::Rank<T>::value
                   == Traits::Tensor::Rank<U>::value)
        struct IsRelative<T, U> : public std::true_type
        {
        };
}  // namespace detail

template<class T, class U, Math::Type::Long... Sizes>
concept Relative
    = Tensor<T> && Tensor<U> && detail::IsRelative<T, U, Sizes...>::value;
template<class T, class U, Math::Type::Long... Sizes>
concept NotRelative = !Relative<T, U, Sizes...>;
template<class T, class U>
concept GreaterRelative
    = Relative<T, U> && detail::IsGreaterRelative<T, U>::value;
template<class T, class U>
concept NotGreaterRelative = !GreaterRelative<T, U>;
template<class T, class U>
concept LesserRelative
    = Relative<T, U> && detail::IsLesserRelative<T, U>::value;
template<class T, class U>
concept NotLesserRelative = !LesserRelative<T, U>;
template<class T, class U>
concept IdenticalRelative
    = Relative<T, U> && detail::IsIdenticalRelative<T, U>::value;
template<class T, class U>
concept NotIdenticalRelative = !IdenticalRelative<T, U>;
namespace detail
{
template<class T, class U>
struct IsSubTensor : public std::false_type
{
};
template<UD::Concepts::Tensor::Tensor T, UD::Concepts::Tensor::Tensor U>
  requires NotRelative<T, U>
struct IsSubTensor<T, U>
    : public IsSubTensor<T, typename Traits::Tensor::Element<U>::type>
{
};
template<UD::Concepts::Tensor::Tensor T, UD::Concepts::Tensor::Tensor U>
  requires Relative<T, U> && NotBaseTensor<T> && NotBaseTensor<U> &&(
      Traits::Tensor::Size<T>::value
      <= Traits::Tensor::Size<U>::value) struct IsSubTensor<T, U>
      : public IsSubTensor<typename Traits::Tensor::Element<T>::type,
                           typename Traits::Tensor::Element<U>::type>
  {
  };
  template<UD::Concepts::Tensor::BaseTensor T,
           UD::Concepts::Tensor::BaseTensor U>
    requires Relative<T, U> &&(
        Traits::Tensor::Size<T>::value
        <= Traits::Tensor::Size<U>::value) struct IsSubTensor<T, U>
        : public std::true_type
    {
    };

    template<class T, class U>
    struct IsSuperTensor : public std::false_type
    {
    };
    template<UD::Concepts::Tensor::Tensor T, UD::Concepts::Tensor::Tensor U>
      requires NotRelative<T, U>
    struct IsSuperTensor<T, U>
        : public IsSuperTensor<typename Traits::Tensor::Element<T>::type, U>
    {
    };
    template<UD::Concepts::Tensor::Tensor T, UD::Concepts::Tensor::Tensor U>
      requires Relative<T, U> && NotBaseTensor<T> && NotBaseTensor<U> &&(
          Traits::Tensor::Size<T>::value
          >= Traits::Tensor::Size<U>::value) struct IsSuperTensor<T, U>
          : public IsSuperTensor<typename Traits::Tensor::Element<T>::type,
                                 typename Traits::Tensor::Element<U>::type>
      {
      };
      template<UD::Concepts::Tensor::BaseTensor T,
               UD::Concepts::Tensor::BaseTensor U>
        requires Relative<T, U> &&(
            Traits::Tensor::Size<T>::value
            >= Traits::Tensor::Size<U>::value) struct IsSuperTensor<T, U>
            : public std::true_type
        {
        };
}  // namespace detail
template<class T, class U>
concept SubTensor = Tensor<T> && Tensor<U> && detail::IsSubTensor<T, U>::value;
template<class T, class U>
concept SuperTensor
    = Tensor<T> && Tensor<U> && detail::IsSuperTensor<T, U>::value;
}  // namespace UD::Concepts::Tensor
namespace UD::Concepts::Tensor
{
namespace detail
{
template<class T, auto S, auto... Ss>
struct OfSizeHelper : public std::false_type
{
};
template<class T, auto S, auto... Ss>
  requires(Traits::Tensor::Size<T>::value == S)
struct OfSizeHelper<T, S, Ss...>
    : public OfSizeHelper<typename Traits::Tensor::Element<T>::type, Ss...>
{
};
template<class T, auto S>
  requires(Traits::Tensor::Size<T>::value == S)
struct OfSizeHelper<T, S> : public std::true_type
{
};
}  // namespace detail
template<class T, auto S>
concept RangeOfSize
    = Range<T> && std::convertible_to<decltype(S), Math::Type::ULong> && detail::
          OfSizeHelper<T, S>::value;
template<class T, auto S, auto... Ss>
concept OfSize
    = Tensor<T> && Traits::Tensor::Rank<T>::value >= sizeof...(Ss) + 1
   && std::conjunction_v<
          std::is_convertible<decltype(S), Math::Type::ULong>,
          std::is_convertible<decltype(Ss), Math::Type::ULong>...>&& detail::
          OfSizeHelper<T, S, Ss...>::value;
}  // namespace UD::Concepts::Tensor
namespace UD::Traits::Tensor
{
template<class T, class Element>
struct FlatSizeOf;

template<Concepts::Tensor::CSizedRange T, class Element>
  requires Concepts::Tensor::RangeOf<T, Element>
struct FlatSizeOf<T, Element>
{
  static consteval Math::Type::ULong test()
  {
    return UD::Traits::Tensor::Size<T>::value
         * FlatSizeOf<typename UD::Traits::Tensor::Element<T>::type,
                      Element>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T, class Element>
  requires Concepts::Tensor::SimpleRangeOf<T, Element>
struct FlatSizeOf<T, Element>
{
  static consteval Math::Type::ULong test()
  {
    return UD::Traits::Tensor::Size<T>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T, class Element>
  requires Concepts::Tensor::TensorOf<T, Element>
struct FlatSizeOf<T, Element>
{
  static consteval Math::Type::ULong test()
  {
    return UD::Traits::Tensor::Size<T>::value
         * FlatSizeOf<typename UD::Traits::Tensor::Element<T>::type,
                      Element>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T, class Element>
  requires Concepts::Tensor::SimpleTensorOf<T, Element>
struct FlatSizeOf<T, Element>
{
  static consteval Math::Type::ULong test()
  {
    return UD::Traits::Tensor::Size<T>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T>
struct FlatSizeOf<T, T>
{
  static consteval Math::Type::ULong test()
  {
    return 1;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T>
struct FlatSize
{
  static consteval Math::Type::ULong test()
  {
    return FlatSizeOf<T,
                      typename UD::Traits::Tensor::BaseElement<T>::type>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
template<Concepts::Tensor::CSizedRange T>
  requires Concepts::Tensor::Tensor<T>
struct FlatSize<T>
{
  static consteval Math::Type::ULong test()
  {
    return FlatSizeOf<T,
                      typename UD::Traits::Tensor::BaseElement<
                          T,
                          UD::Traits::Tensor::SatisfiesTensor>::type>::value;
  }
  static constexpr Math::Type::ULong value = test();
};
}  // namespace UD::Traits::Tensor
namespace UD::Tensor
{
namespace Concepts = UD::Concepts::Tensor;
namespace Traits   = UD::Traits::Tensor;

namespace detail
{

template<Concepts::Arithmeticable _T,
         UD::Math::Type::ULong    _Size,
         UD::Math::Type::ULong... _Sizes>
struct RecursiveTensorHelper
{
  using type
      = TensorRaw<typename RecursiveTensorHelper<_T, _Sizes...>::type, _Size>;
};
template<Concepts::Arithmeticable _T, UD::Math::Type::ULong _Size>
struct RecursiveTensorHelper<_T, _Size>
{
  using type = TensorRaw<_T, _Size>;
};
}  // namespace detail

template<class _T, UD::Math::Type::ULong _Size>
using Tensor_ = TensorRaw<_T, _Size>;
template<class _T, UD::Math::Type::ULong _N>
using Vector_ = TensorRaw<_T, _N>;
template<class _T, UD::Math::Type::ULong _M, UD::Math::Type::ULong _N>
using Matrix_ = TensorRaw<Vector_<_T, _N>, _M>;

template<class _T, UD::Math::Type::ULong _Size, UD::Math::Type::ULong... _Sizes>
using Tensor =
    typename detail::RecursiveTensorHelper<_T, _Size, _Sizes...>::type;
template<class _T, UD::Math::Type::ULong _N>
using Vector = TensorRaw<_T, _N>;
template<class _T, UD::Math::Type::ULong _M, UD::Math::Type::ULong _N>
using Matrix = TensorRaw<Vector<_T, _N>, _M>;
}  // namespace UD::Tensor
namespace UD::Tensor
{
template<Concepts::Tensor To>
auto Convert(const Concepts::Tensor auto& from) -> To
    requires(Traits::Rank<To>::value
             > Traits::Rank<std::remove_cvref_t<decltype(from)>>::value)
{
  To to{};
  *std::ranges::begin(to) = Convert<std::ranges::range_value_t<To>>(from);
  return std::move(to);
}
template<Concepts::Tensor To>
auto Convert(const Concepts::Tensor auto& from) -> To
    requires(Traits::Rank<To>::value
             < Traits::Rank<std::remove_cvref_t<decltype(from)>>::value)
{
  return Convert<To>(*std::ranges::begin(from));
}
template<Concepts::Tensor To>
    auto Convert(const Concepts::Tensor auto& from) -> To requires
    Concepts::CSizedRange<To> && Concepts::CSizedRange<std::remove_cvref_t<decltype(from)>> &&(
        Concepts::NotBaseVector<
            To> || Concepts::NotBaseVector<std::remove_cvref_t<decltype(from)>>)&&(Traits::
                                                                                       Rank<
                                                                                           To>::
                                                                                           value
                                                                                   == Traits::Rank<
                                                                                       std::remove_cvref_t<
                                                                                           decltype(from)>>::
                                                                                       value)
    && (UD::Traits::Tensor::Size<To>::value
        >= UD::Traits::Tensor::Size<std::remove_cvref_t<decltype(from)>>::value)
{
  To to{};
  for (auto [to_it, from_it]
       = std::make_pair(std::ranges::begin(to), std::ranges::begin(from));
       to_it != std::ranges::end(to) && from_it != std::ranges::end(from);
       ++to_it, ++from_it)
  {
    *to_it = Convert<std::ranges::range_value_t<To>>(*from_it);
  }
  return std::move(to);
}
template<Concepts::BaseVector To>
auto Convert(const Concepts::BaseVector auto&
                 from) -> To requires Concepts::CSizedRange<To> && Concepts::
    CSizedRange<std::remove_cvref_t<decltype(from)>> &&(
        UD::Traits::Tensor::Size<To>::value
        >= UD::Traits::Tensor::Size<std::remove_cvref_t<decltype(from)>>::value)
{
  To to{};
  for (auto [to_it, from_it]
       = std::make_pair(std::ranges::begin(to), std::ranges::begin(from));
       to_it != std::ranges::end(to) && from_it != std::ranges::end(from);
       ++to_it, ++from_it)
  {
    *to_it = *from_it;
  }
  return std::move(to);
}
template<Concepts::Tensor To>
auto CastConvert(const Concepts::Tensor auto& from) -> To
    requires(Traits::Rank<To>::value
             > Traits::Rank<std::remove_cvref_t<decltype(from)>>::value)
{
  To to{};
  *std::ranges::begin(to) = CastConvert<std::ranges::range_value_t<To>>(from);
  return std::move(to);
}
template<Concepts::Tensor To>
auto CastConvert(const Concepts::Tensor auto& from) -> To
    requires(Traits::Rank<To>::value
             < Traits::Rank<std::remove_cvref_t<decltype(from)>>::value)
{
  return CastConvert<To>(*std::ranges::begin(from));
}
template<Concepts::Tensor To>
    auto CastConvert(const Concepts::Tensor auto& from) -> To requires(
        Concepts::NotBaseVector<
            To> || Concepts::NotBaseVector<std::remove_cvref_t<decltype(from)>>)
    && (Traits::Rank<To>::value
        == Traits::Rank<std::remove_cvref_t<decltype(from)>>::value)
{
  To to{};
  for (auto [to_it, from_it]
       = std::make_pair(std::ranges::begin(to), std::ranges::begin(from));
       to_it != std::ranges::end(to) && from_it != std::ranges::end(from);
       ++to_it, ++from_it)
  {
    *to_it = CastConvert<std::ranges::range_value_t<To>>(*from_it);
  }
  return std::move(to);
}
template<Concepts::BaseVector To>
auto CastConvert(const Concepts::BaseVector auto& from) -> To
{
  To to{};
  for (auto [to_it, from_it]
       = std::make_pair(std::ranges::begin(to), std::ranges::begin(from));
       to_it != std::ranges::end(to) && from_it != std::ranges::end(from);
       ++to_it, ++from_it)
  {
    *to_it = static_cast<std::ranges::range_value_t<To>>(*from_it);
  }
  return std::move(to);
}

template<Concepts::Tensor T, Concepts::Arithmeticable U>
  requires Concepts::SimpleTensorOf<T, U>
constexpr void Fill(T& t, const U& u)
{
  std::ranges::fill(t, u);
}
template<Concepts::Tensor T, Concepts::Arithmeticable U>
  requires Concepts::TensorOf<T, U>
constexpr void Fill(T& t, const U& u)
{
  for (auto& v : t)
  {
    Fill(v, u);
  }
}
template<Concepts::Tensor T, Concepts::Arithmeticable U>
  requires Concepts::NotTensorOf<T, U> && Concepts::
      SuperTensor<T, U> && Concepts::Relative<T, U>
constexpr void Fill(T& t, const U& u)
{
  t = u;
}
template<Concepts::Tensor T, Concepts::Arithmeticable U>
  requires Concepts::NotTensorOf<T, U> && Concepts::
      SuperTensor<T, U> && Concepts::NotRelative<T, U>
constexpr void Fill(T& t, const U& u)
{
  for (auto& v : t)
  {
    Fill(v, u);
  }
}
}  // namespace UD::Tensor
namespace UD::Tensor
{
template<Concepts::Arithmeticable _T, UD::Math::Type::ULong _Size>
struct TensorRaw
{
  using Type          = _T;
  using ContainerType = std::array<_T, _Size>;

  ContainerType _data;

  // TensorRaw(std::initializer_list<_T> list) : _data{list} {}
  constexpr auto size() const noexcept requires requires(ContainerType c)
  {
    {c.size()};
  }
  {
    return _data.size();
  }
  constexpr auto sizes() const noexcept requires requires(ContainerType c)
  {
    {c.begin()->sizes()};
  }
  {
    return std::tuple_cat(std::make_tuple(size()), _data.begin()->sizes());
  }
  constexpr auto sizes() const noexcept
      requires(!requires(ContainerType c) { {c.begin()->sizes()}; })
  {
    return std::make_tuple(size());
  }
  auto begin() requires requires(ContainerType c)
  {
    {c.begin()};
  }
  {
    return _data.begin();
  }
  auto begin() const requires requires(const ContainerType c)
  {
    {c.begin()};
  }
  {
    return _data.begin();
  }
  auto rbegin() requires requires(ContainerType c)
  {
    {c.rbegin()};
  }
  {
    return _data.rbegin();
  }
  auto rbegin() const requires requires(const ContainerType c)
  {
    {c.rbegin()};
  }
  {
    return _data.rbegin();
  }
  auto end() requires requires(ContainerType c)
  {
    {c.end()};
  }
  {
    return _data.end();
  }
  auto end() const requires requires(const ContainerType c)
  {
    {c.end()};
  }
  {
    return _data.end();
  }
  auto rend() requires requires(ContainerType c)
  {
    {c.rend()};
  }
  {
    return _data.rend();
  }
  auto rend() const requires requires(const ContainerType c)
  {
    {c.rend()};
  }
  {
    return _data.rend();
  }
  auto cbegin() requires requires(ContainerType c)
  {
    {c.cbegin()};
  }
  {
    return _data.cbegin();
  }
  auto cbegin() const requires requires(const ContainerType c)
  {
    {c.cbegin()};
  }
  {
    return _data.cbegin();
  }
  auto crbegin() requires requires(ContainerType c)
  {
    {c.crbegin()};
  }
  {
    return _data.crbegin();
  }
  auto crbegin() const requires requires(const ContainerType c)
  {
    {c.crbegin()};
  }
  {
    return _data.crbegin();
  }
  auto cend() requires requires(ContainerType c)
  {
    {c.cend()};
  }
  {
    return _data.cend();
  }
  auto cend() const requires requires(const ContainerType c)
  {
    {c.cend()};
  }
  {
    return _data.cend();
  }
  auto crend() requires requires(ContainerType c)
  {
    {c.crend()};
  }
  {
    return _data.crend();
  }
  auto crend() const requires requires(const ContainerType c)
  {
    {c.crend()};
  }
  {
    return _data.crend();
  }
  auto push_back(_T&& t) requires requires(_T t, ContainerType c)
  {
    {c.push_back(t)};
  }
  {
    return _data.push_back(std::forward<_T>(t));
  }
  // template<class T>
  // auto& at()
  //{
  //   return this->_data.at(T::value);
  // }
  // template<class T>
  //   requires requires()
  //   {
  //     typename T::Next;
  //   }
  // auto& at()
  //{
  //   return this->_data.at(T::value).template at<typename T::Next>();
  // }
  // template<class T>
  // auto at() const
  //{
  //   return this->_data.at(T::value);
  // }
  // template<class T>
  //   requires requires()
  //   {
  //     typename T::Next;
  //   }
  // auto at() const
  //{
  //   return this->_data.at(T::value).template at<typename T::Next>();
  // }
  // template<class S, class... Ss>
  // struct AtHelper
  //{
  //  static auto& Call(TensorRaw& t, const S& s, const Ss&... ss)
  //  {
  //    return std::remove_cvref_t<decltype(t._data.at(s))>::AtHelper::Call(
  //        t._data.at(s),
  //        ss...);
  //  }
  //  static auto Call(const TensorRaw& t, const S& s, const Ss&... ss)
  //  {
  //    return std::remove_cvref_t<decltype(t._data.at(s))>::AtHelper::Call(
  //        t._data.at(s),
  //        ss...);
  //  }
  //};
  // template<class S>
  // struct AtHelper<S>
  //{
  //  static auto& Call(TensorRaw& t, const S& s)
  //  {
  //    return t._data.at(s);
  //  }
  //  static auto Call(const TensorRaw& t, const S& s)
  //  {
  //    return t._data.at(s);
  //  }
  //};
  // template<Math::Type::ULong S, Math::Type::ULong... Ss>
  // struct AtHelper2
  //{
  //  static auto& Call(TensorRaw& t)
  //  {
  //    return std::remove_cvref_t<decltype(t._data.at(S))>::AtHelper::Call(
  //        t._data.at(S),
  //        Ss...);
  //  }
  //  static auto Call(const TensorRaw& t)
  //  {
  //    return std::remove_cvref_t<decltype(t._data.at(S))>::AtHelper::Call(
  //        t._data.at(S),
  //        Ss...);
  //  }
  //};
  // template<Math::Type::ULong S>
  // struct AtHelper2<S>
  //{
  //  static auto& Call(TensorRaw& t)
  //  {
  //    return t._data.at(S);
  //  }
  //  static auto Call(const TensorRaw& t)
  //  {
  //    return t._data.at(S);
  //  }
  //};
  // template<Math::Type::ULong S, Math::Type::ULong... Ss>
  // auto& at()
  //{
  //  return AtHelper2<S, Ss...>::Call(*this);
  //}
  // template<Math::Type::ULong S, Math::Type::ULong... Ss>
  // auto at() const
  //{
  //  return AtHelper2<S, Ss...>::Call(*this);
  //}
  // template<class S, class... Ss>
  // auto& at(S s, Ss... ss)
  //{
  //  return AtHelper<S, Ss...>::Call(*this, s, ss...);
  //}
  // template<class S, class... Ss>
  // auto at(S s, Ss... ss) const
  //{
  //  return AtHelper<S, Ss...>::Call(*this, s, ss...);
  //}

  template<Math::Type::ULong S, Math::Type::ULong... Ss>
    requires(sizeof...(Ss) > 0)
  auto at() -> auto&
  {
    return this->_data.at(S).template at<Ss...>();
  }
  template<Math::Type::ULong S, Math::Type::ULong... Ss>
    requires(sizeof...(Ss) == 0)
  auto at() -> _T&
  {
    return this->_data.at(S);
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) > 0)
  auto at(S&& s, Ss&&... ss) -> auto&
  {
    return this->_data.at(std::forward<S>(s)).at(std::forward<Ss>(ss)...);
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) == 0)
  auto at(S index) -> _T&
  {
    return this->_data.at(index);
  }

  template<Math::Type::ULong S, Math::Type::ULong... Ss>
    requires(sizeof...(Ss) > 0)
  auto at() const
  {
    return this->_data.at(S).template at<Ss...>();
  }
  template<Math::Type::ULong S, Math::Type::ULong... Ss>
    requires(sizeof...(Ss) == 0)
  auto at() const
  {
    return this->_data.at(S);
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) > 0)
  auto at(S&& s, Ss&&... ss) const
  {
    auto t = this->_data.at(std::forward<S>(s));
    return t.at(std::forward<Ss>(ss)...);
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) == 0)
  auto at(S index) const
  {
    return this->_data.at(index);
  }

  template<std::convertible_to<Math::Type::ULong> S>
  auto& rat(S&& s)
  {
    return this->at(std::forward<S>(s));
  }
  template<std::convertible_to<Math::Type::ULong> S>
  auto rat(S&& s) const
  {
    return this->at(std::forward<S>(s));
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) > 0)
  auto& rat(S&& s, Ss&&... ss)
  {
    return this->rat(std::forward<Ss>(ss)...).at(std::forward<S>(s));
  }
  template<std::convertible_to<Math::Type::ULong> S,
           std::convertible_to<Math::Type::ULong>... Ss>
    requires(sizeof...(Ss) > 0)
  auto rat(S&& s, Ss&&... ss) const
  {
    return this->rat(std::forward<Ss>(ss)...).at(std::forward<S>(s));
  }

  auto& operator[](UD::Math::Type::ULong index)
  {
    return this->_data[index];
  }
  auto operator[](UD::Math::Type::ULong index) const
  {
    return this->_data[index];
  }
  friend std::ostream& operator<<(std::ostream& os, const TensorRaw& a) requires
      requires(std::ostream& os, _T t)
  {
    {
      os << t
      } -> std::same_as<std::ostream&>;
  }
  {
    auto it = a.begin();
    os << '[';
    if (it == a.end())
    {
      os << ']';
      return os;
    }
    os << *it;
    ++it;
    for (; it != a.end(); ++it)
    {
      os << ',' << *it;
    }
    return os << ']';
  }

  template<std::convertible_to<TensorRaw> U>
  TensorRaw& operator+=(U&& rhs)
  {
    return UD::Math::zip_transform(
        *this,
        std::forward<U>(rhs),
        [](auto& a, auto&& b)
        {
          return a += std::forward<std::remove_reference_t<decltype(b)>>(b);
        });
  }
  template<class U>
    requires(!std::convertible_to<U, TensorRaw>)
  &&std::convertible_to<U, typename Traits::BaseElement<TensorRaw>::type>
      TensorRaw& operator+=(U&& rhs)
  {
    TensorRaw ret;
    Fill(ret, std::forward<typename Traits::BaseElement<TensorRaw>::type>(rhs));
    return UD::Math::zip_transform(
        *this,
        std::move(ret),
        [](auto& a, auto&& b)
        {
          return a += std::forward<std::remove_reference_t<decltype(b)>>(b);
        });
  }
  template<std::convertible_to<TensorRaw> V>
  friend TensorRaw operator+(TensorRaw lhs, V&& rhs)
  {
    return UD::Math::zip_transform(lhs,
                                   std::forward<V>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          + std::forward<decltype(b)>(b);
                                   });
  }
  template<std::same_as<TensorRaw> T, class V>
    requires((!std::convertible_to<V, T>)&&std::
                 convertible_to<V, typename Traits::BaseElement<T>::type>)
  friend T operator+(T lhs, V&& rhs)
  {
    T ret;
    Fill(ret, std::forward<typename Traits::BaseElement<T>::type>(rhs));
    return UD::Math::zip_transform(lhs,
                                   std::move(ret),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          + std::forward<decltype(b)>(b);
                                   });
  }

  template<std::convertible_to<TensorRaw> U>
  TensorRaw& operator-=(U&& rhs)
  {
    return UD::Math::zip_transform(*this,
                                   std::forward<U>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                         -= std::forward<decltype(b)>(b);
                                   });
  }
  template<class U>
    requires(!std::convertible_to<U, TensorRaw>)
  &&std::convertible_to<U, typename Traits::BaseElement<TensorRaw>::type>
      TensorRaw& operator-=(U&& rhs)
  {
    TensorRaw ret;
    Fill(ret, std::forward<typename Traits::BaseElement<TensorRaw>::type>(rhs));
    return UD::Math::zip_transform(
        *this,
        std::move(ret),
        [](auto& a, auto&& b)
        {
          return a -= std::forward<std::remove_reference_t<decltype(b)>>(b);
        });
  }
  template<std::convertible_to<TensorRaw> V>
  friend TensorRaw operator-(TensorRaw lhs, V&& rhs)
  {
    return UD::Math::zip_transform(lhs,
                                   std::forward<V>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          - std::forward<decltype(b)>(b);
                                   });
  }
  template<std::same_as<TensorRaw> T, class V>
    requires((!std::convertible_to<V, T>)&&std::
                 convertible_to<V, typename Traits::BaseElement<T>::type>)
  friend T operator-(T lhs, V&& rhs)
  {
    T ret;
    Fill(ret, std::forward<typename Traits::BaseElement<T>::type>(rhs));
    return UD::Math::zip_transform(lhs,
                                   std::move(ret),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          - std::forward<decltype(b)>(b);
                                   });
  }

  template<std::convertible_to<TensorRaw> U>
  TensorRaw& operator*=(U&& rhs)
  {
    return UD::Math::zip_transform(*this,
                                   std::forward<U>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                         *= std::forward<decltype(b)>(b);
                                   });
  }
  template<class U>
    requires(!std::convertible_to<U, TensorRaw>)
  &&std::convertible_to<U, typename Traits::BaseElement<TensorRaw>::type>
      TensorRaw& operator*=(U&& rhs)
  {
    TensorRaw ret;
    Fill(ret, std::forward<typename Traits::BaseElement<TensorRaw>::type>(rhs));
    return UD::Math::zip_transform(
        *this,
        std::move(ret),
        [](auto& a, auto&& b)
        {
          return a *= std::forward<std::remove_reference_t<decltype(b)>>(b);
        });
  }
  template<std::convertible_to<TensorRaw> V>
  friend TensorRaw operator*(TensorRaw lhs, V&& rhs)
  {
    return UD::Math::zip_transform(lhs,
                                   std::forward<V>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          * std::forward<decltype(b)>(b);
                                   });
  }
  template<std::same_as<TensorRaw> T, class V>
    requires((!std::convertible_to<V, T>)&&std::
                 convertible_to<V, typename Traits::BaseElement<T>::type>)
  friend T operator*(T lhs, V&& rhs)
  {
    T ret;
    Fill(ret, std::forward<typename Traits::BaseElement<T>::type>(rhs));
    return UD::Math::zip_transform(lhs,
                                   std::move(ret),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          * std::forward<decltype(b)>(b);
                                   });
  }

  template<std::convertible_to<TensorRaw> U>
  TensorRaw& operator/=(U&& rhs)
  {
    return UD::Math::zip_transform(*this,
                                   std::forward<U>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                         /= std::forward<decltype(b)>(b);
                                   });
  }
  template<class U>
    requires(!std::convertible_to<U, TensorRaw>)
  &&std::convertible_to<U, typename Traits::BaseElement<TensorRaw>::type>
      TensorRaw& operator/=(U&& rhs)
  {
    TensorRaw ret;
    Fill(ret, std::forward<typename Traits::BaseElement<TensorRaw>::type>(rhs));
    return UD::Math::zip_transform(
        *this,
        std::move(ret),
        [](auto& a, auto&& b)
        {
          return a /= std::forward<std::remove_reference_t<decltype(b)>>(b);
        });
  }
  template<std::convertible_to<TensorRaw> V>
  friend TensorRaw operator/(TensorRaw lhs, V&& rhs)
  {
    return UD::Math::zip_transform(lhs,
                                   std::forward<V>(rhs),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          / std::forward<decltype(b)>(b);
                                   });
  }
  template<std::same_as<TensorRaw> T, class V>
    requires((!std::convertible_to<V, T>)&&std::
                 convertible_to<V, typename Traits::BaseElement<T>::type>)
  friend T operator/(T lhs, V&& rhs)
  {
    T ret;
    Fill(ret, std::forward<typename Traits::BaseElement<T>::type>(rhs));
    return UD::Math::zip_transform(lhs,
                                   std::move(ret),
                                   [](auto&& a, auto&& b)
                                   {
                                     return std::forward<decltype(a)>(a)
                                          / std::forward<decltype(b)>(b);
                                   });
  }
  TensorRaw operator-() const
  {
    return *this * -1;
  }

  // implicit cast
  template<Concepts::Tensor T>
    requires Concepts::CSizedRange<T> &&(
        (UD::Concepts::NotOnlyExplicitlyConvertibleTo<
             TensorRaw,
             std::ranges::range_value_t<
                 T>> && (Traits::Rank<T>::value > Traits::Rank<std::ranges::range_value_t<TensorRaw>>::value + 1))
        || ((Traits::Rank<T>::value
             == Traits::Rank<std::ranges::range_value_t<TensorRaw>>::value + 1)
            && (UD::Traits::Tensor::Size<TensorRaw>::value
                <= UD::Traits::Tensor::Size<T>::value)
            && UD::Concepts::NotOnlyExplicitlyConvertibleTo<
                std::ranges::range_value_t<TensorRaw>,
                std::ranges::range_value_t<T>>))
        operator T() const
    {
      return Convert<T>(*this);
    }
    // explicit cast
    template<Concepts::Tensor T>
      requires(
          Concepts::CSizedRange<
              T> && (UD::Concepts::OnlyExplicitlyConvertibleTo<TensorRaw, std::ranges::range_value_t<T>> && (Traits::Rank<T>::value > Traits::Rank<std::ranges::range_value_t<TensorRaw>>::value + 1))
          || (Traits::Rank<T>::value
              < Traits::Rank<std::ranges::range_value_t<TensorRaw>>::value + 1)
          || ((Traits::Rank<T>::value
               == Traits::Rank<std::ranges::range_value_t<TensorRaw>>::value
                      + 1)
              && ((UD::Traits::Tensor::Size<TensorRaw>::value
                   > UD::Traits::Tensor::Size<T>::value)
                  || (UD::Concepts::OnlyExplicitlyConvertibleTo<
                          std::ranges::range_value_t<TensorRaw>,
                          std::ranges::range_value_t<
                              T>> && UD::Traits::Tensor::Size<TensorRaw>::value <= UD::Traits::Tensor::Size<T>::value))))
    explicit operator T() const
    {
      return CastConvert<T>(*this);
    }

    auto& x() requires(Traits::Size<TensorRaw>::value >= 1)
    {
      return this->at(0);
    }
    auto& y() requires(Traits::Size<TensorRaw>::value >= 2)
    {
      return this->at(1);
    }
    auto& z() requires(Traits::Size<TensorRaw>::value >= 3)
    {
      return this->at(2);
    }
};
template<class T, std::same_as<T>... Ts>
TensorRaw(T, Ts...) -> TensorRaw<T, sizeof...(Ts) + 1>;
static_assert(Concepts::Tensor<TensorRaw<int, 2>>);

template<Concepts::Arithmeticable To>
constexpr auto Flatten(const Concepts::TensorOf<To> auto& from) requires
    Concepts::CSizedRange<std::remove_cvref_t<decltype(from)>>
{
  using RetType = Tensor<
      To,
      Traits::FlatSizeOf<std::remove_cvref_t<decltype(from)>, To>::value>;
  auto ret = RetType{};
  Flatten<To>(from, std::ranges::begin(ret));
  return std::move(ret);
}
constexpr auto Flatten(const Concepts::Tensor auto& from)
{
  return Flatten<
      typename Traits::BaseElement<std::remove_cvref_t<decltype(from)>,
                                   Traits::SatisfiesTensor>::type>(from);
}

template<Concepts::Arithmeticable To>
constexpr auto Flatten(const Concepts::TensorOf<To> auto& from, auto it)
{
  for (auto& val : from)
  {
    it = Flatten<To>(val, it);
  }
  return std::move(it);
}
template<Concepts::Arithmeticable To>
constexpr auto Flatten(const Concepts::VectorOf<To> auto& from, auto it)
{
  for (auto& val : from)
  {
    *it = val;
    ++it;
  }
  return std::move(it);
}

template<Concepts::Tensor L, Concepts::Tensor R>
  requires Concepts::Relative<L, R> && std::convertible_to <
      typename Traits::Element<R>::type,
typename Traits::Element<L>::type
    > auto Merge(const L& lhs, const R& rhs)
          -> Tensor<typename Traits::Element<L>::type,
                    Traits::Size<L>::value + Traits::Size<R>::value>
{
  auto ret
      = static_cast<Tensor<typename Traits::Element<L>::type,
                           Traits::Size<L>::value + Traits::Size<R>::value>>(
          lhs);
  for (auto [it, jt] = std::pair{ret.rbegin(), rhs.rbegin()};
       it != ret.rend() && jt != rhs.rend();
       ++it, ++jt)
  {
    *it = *jt;
  }
  return std::move(ret);
}
template<Concepts::Tensor L, Concepts::Tensor R>
  requires Concepts::NotRelative<L, R> &&(
      Traits::Rank<L>::value < Traits::Rank<R>::value) auto Merge(const L& lhs,
                                                                  const R& rhs)
  {
    auto ret
        = Tensor<decltype(Merge(lhs, *rhs.begin())), Traits::Size<R>::value>{};
    for (auto [it, jt] = std::pair{ret.begin(), rhs.begin()};
         it != ret.end() && jt != rhs.end();
         ++it, ++jt)
    {
      *it = Merge(lhs, *jt);
    }
    return std::move(ret);
  }
  template<Concepts::Tensor L, Concepts::Tensor R>
    requires Concepts::NotRelative<L, R> &&(
        Traits::Rank<L>::value
        > Traits::Rank<R>::value) auto Merge(const L& lhs, const R& rhs)
    {
      auto ret = Tensor<decltype(Merge(*lhs.begin(), rhs)),
                        Traits::Size<L>::value>{};
      for (auto [it, jt] = std::pair{ret.begin(), lhs.begin()};
           it != ret.end() && jt != lhs.end();
           ++it, ++jt)
      {
        *it = Merge(*jt, rhs);
      }
      return std::move(ret);
    }
    template<Concepts::Tensor T>
    auto Minor(const T& tensor, const Math::Type::ULong& index)
        -> Concepts::Relative<T, -1> auto
    {
      auto ret = Tensor<typename Traits::Element<T>::type,
                        Traits::Size<T>::value - 1>{};
      auto it  = ret.begin();
      auto jt  = tensor.begin();
      auto i   = UD::Math::Type::ULong{0};
      for (; i < index; ++i, ++it, ++jt)
      {
        *it = *jt;
        if (it == ret.end() || jt == tensor.end())
          throw std::out_of_range("Index out of bounds");
      }
      ++jt;
      for (; it != ret.end() && jt != tensor.end(); ++it, ++jt)
      {
        *it = *jt;
      }
      return ret;
    }
    template<Concepts::Tensor T,
             std::convertible_to<Math::Type::ULong>... Indices>
    auto Minor(const T&                 tensor,
               const Math::Type::ULong& index,
               Indices... indices) -> Concepts::Relative<T, -1> auto
    {
      auto ret = Tensor<decltype(Minor(*tensor.begin(), indices...)),
                        Traits::Size<T>::value - 1>{};
      auto it  = ret.begin();
      auto jt  = tensor.begin();
      auto i   = UD::Math::Type::ULong{0};
      for (; i < index; ++i, ++it, ++jt)
      {
        *it = Minor(*jt, indices...);
        if (it == ret.end() || jt == tensor.end())
          throw std::out_of_range("Index out of bounds");
      }
      ++jt;
      for (; it != ret.end() && jt != tensor.end(); ++it, ++jt)
      {
        *it = Minor(*jt, indices...);
      }
      return ret;
    }
    namespace detail
    {
    template<class T>
    struct PackToTensor;
    template<auto _V, auto... _Vs>
    struct PackToTensor<Pack::ValuesPack<_V, _Vs...>>
    {
      template<class T>
      using Type = Tensor<T, _V, _Vs...>;
    };
    }  // namespace detail
    // template<class T, Math::Type::ULong... Ss>
    // auto Transpose(const Tensor<T, Ss...>& t)
    //{
    // }
    // template<Concepts::Matrix T>
    // auto Transpose(const T& t) -> Concepts::Matrix auto
    //{
    //  auto ret = Tensor<
    //      typename Traits::Element<typename Traits::Element<T>::type>::type,
    //      Traits::Size<typename Traits::Element<T>::type>::value,
    //      Traits::Size<T>::value>{};
    //  for (Math::Type::ULong i = 0; i < Traits::Size<T>::value; ++i)
    //  {
    //    for (Math::Type::ULong j = 0;
    //         j < Traits::Size<typename Traits::Element<T>::type>::value;
    //         ++j)
    //    {
    //      ret[j][i] = t[i][j];
    //    }
    //  }
    //  return ret;
    //}
    namespace detail
    {
    template<class T, class... Ts>
    concept AllConvertibleTo
        = std::conjunction_v<std::is_convertible<Ts, T>...>;
    template<class T,
             class U,
             std::convertible_to<Math::Type::ULong> I,
             std::convertible_to<Math::Type::ULong>... Is>
    void TransposeLambda_ClangFormatIssueWorkaround(T&       to,
                                                    const U& from,
                                                    I        i,
                                                    Is... is)
    {
      to.rat(i, is...) = from.at(std::forward<decltype(i)>(i),
                                 std::forward<decltype(is)>(is)...);
    }
    }  // namespace detail

    /*
     *  Could be greatly simplified through the use of a constrained generic
     * lambda. However, ClangFormat automatically changes file type to
     * Objective-C when such constructs are detected.
     *  TODO: Simplify when ClangFormat is updated.
     */
    template<class T, Math::Type::ULong S, Math::Type::ULong... Ss>
    auto Transpose(const Tensor<T, S, Ss...>& t)
    {
      typename detail::PackToTensor<
          typename Pack::ValuesPack<S, Ss...>::Reverse>::template Type<T>
          ret;
      For<S, Ss...>::Call(
          UD::Tensor::detail::TransposeLambda_ClangFormatIssueWorkaround<
              std::remove_cvref_t<decltype(ret)>,
              std::remove_cvref_t<decltype(t)>,
              decltype(S),
              decltype(Ss)...>,
          ret,
          t);
      return ret;
    }
    // template<class T>
    //   requires Concepts::Square<T>
    // auto Trace(const T& t) ->
    //     typename Traits::Element<typename Traits::Element<T>::type>::type
    //{
    //   using RetType =
    //       typename Traits::Element<typename Traits::Element<T>::type>::type;
    //   auto ret = RetType{};
    //   for (Math::Type::ULong i = 0; i < Traits::Size<T>::value; ++i)
    //   {
    //     ret += t[i][i];
    //   }
    //   return ret;
    // }

    auto TraceIndex(const auto& t, UD::Math::Type::ULong i)  requires
        Concepts::Square<std::remove_cvref_t<decltype(t)>>
    {
      return TraceIndex(t.at(i), i);
    }
    auto TraceIndex(const auto& t, UD::Math::Type::ULong i) requires
        Concepts::Square<std::remove_cvref_t<decltype(t)>> && Concepts::BaseTensor<std::remove_cvref_t<decltype(t)>>
    {
      return t.at(i);
    }
    auto Trace(const auto& t) requires
        Concepts::Square<std::remove_cvref_t<decltype(t)>>
    {
      auto ret = typename Traits::BaseElement<std::remove_cvref_t<decltype(t)>>::type{};
      for (Math::Type::ULong i = 0; i < Traits::Size<std::remove_cvref_t<decltype(t)>>::value; ++i)
      {
        ret += TraceIndex(t,i);
      }
      return ret;
    }
    template<class I,
             class J,
             class T,
             class U,
             Math::Type::ULong M,
             Math::Type::ULong N>
      requires(I::Value < M)
    &&(J::Value < N)
        && (!I::Next::Empty && !J::Next::Empty) inline auto IndexMultiplication(
            const TensorRaw<T, M>& lhs,
            const TensorRaw<U, N>& rhs)
    {
      return IndexMultiplication<typename I::Next, typename J::Next>(
          lhs[I::Value],
          rhs[J::Value]);
    }
    template<class I,
             class J,
             class T,
             class U,
             Math::Type::ULong M,
             Math::Type::ULong N>
      requires(I::Value < M)
    &&(J::Value < N)
        && (I::Next::Empty || J::Next::Empty) inline auto IndexMultiplication(
               const TensorRaw<T, M>& lhs,
               const TensorRaw<U, N>& rhs) -> T
    {
      return lhs[I::Value] * rhs[J::Value];
    }
    namespace detail
    {
    template<class T>
    struct TrailingZero : public std::false_type
    {
    };
    template<auto... Ts>
      requires std::conjunction_v<std::is_convertible<
          decltype(Ts),
          Math::Type::ULong>...> && TrailingZero<typename Pack::
                                                     ValuesPack<Ts...>::Next>::
          value &&(Pack::ValuesPack<Ts...>::Value
                   == 0) struct TrailingZero<Pack::ValuesPack<Ts...>>
          : public std::true_type
      {
      };
      template<auto T>
        requires std::convertible_to<decltype(T), Math::Type::ULong> &&(
            Pack::ValuesPack<T>::Value
            == 0) struct TrailingZero<Pack::ValuesPack<T>>
            : public std::true_type
        {
        };
        template<class T>
        struct LastNonZero : public std::false_type
        {
        };
        template<auto... Ts>
          requires std::conjunction_v<std::is_convertible<
              decltype(Ts),
              Math::Type::ULong>...> && TrailingZero<typename Pack::
                                                         ValuesPack<
                                                             Ts...>::Next>::
              value &&(!TrailingZero<Pack::ValuesPack<Ts...>>::
                           value) struct LastNonZero<Pack::ValuesPack<Ts...>>
              : public std::true_type
          {
          };
          template<class T>
          concept ValueIsTrailingZero = TrailingZero<T>::value;
          template<class T>
          concept ValueIsLastNonZero = LastNonZero<T>::value;
          template<class Is, class Ss>
          struct Increment;
          template<auto I, auto... Is, auto S, auto... Ss>
            requires AllConvertibleTo<Math::Type::ULong,
                                      decltype(I),
                                      decltype(Is)...,
                                      decltype(S),
                                      decltype(Ss)...>
          struct Increment<Pack::ValuesPack<I, Is...>,
                           Pack::ValuesPack<S, Ss...>>
          {
            using type = typename Pack::ValuesPackConcat<
                Pack::ValuesPack<I>,
                typename Increment<Pack::ValuesPack<Is...>,
                                   Pack::ValuesPack<Ss...>>::type>::type;
          };
          template<auto I, auto... Is, auto S, auto... Ss>
            requires AllConvertibleTo<
                Math::Type::ULong,
                decltype(I),
                decltype(Is)...,
                decltype(S),
                decltype(Ss)...> && ValueIsTrailingZero<typename Increment<Pack::
                                                                               ValuesPack<
                                                                                   Is...>,
                                                                           Pack::ValuesPack<
                                                                               Ss...>>::
                                                            type> &&(I + 1
                                                                     < S) struct
                Increment<Pack::ValuesPack<I, Is...>,
                          Pack::ValuesPack<S, Ss...>>
            {
              using type = typename Pack::ValuesPackConcat<
                  Pack::ValuesPack<I + 1>,
                  typename Increment<Pack::ValuesPack<Is...>,
                                     Pack::ValuesPack<Ss...>>::type>::type;
            };
            template<auto I, auto... Is, auto S, auto... Ss>
              requires AllConvertibleTo<
                  Math::Type::ULong,
                  decltype(I),
                  decltype(Is)...,
                  decltype(S),
                  decltype(Ss)...> && ValueIsTrailingZero<typename Increment<Pack::
                                                                                 ValuesPack<
                                                                                     Is...>,
                                                                             Pack::ValuesPack<
                                                                                 Ss...>>::
                                                              type> &&(I + 1
                                                                       == S) struct
                  Increment<Pack::ValuesPack<I, Is...>,
                            Pack::ValuesPack<S, Ss...>>
              {
                using type = typename Pack::ValuesPackConcat<
                    Pack::ValuesPack<0>,
                    typename Increment<Pack::ValuesPack<Is...>,
                                       Pack::ValuesPack<Ss...>>::type>::type;
              };

              template<auto I, auto S>
                requires AllConvertibleTo<
                    Math::Type::ULong,
                    decltype(I),
                    decltype(S)> &&(I + 1
                                    == S) struct Increment<Pack::ValuesPack<I>,
                                                           Pack::ValuesPack<S>>
                {
                  using type = Pack::ValuesPack<0>;
                };
                template<auto I, auto S>
                  requires AllConvertibleTo<
                      Math::Type::ULong,
                      decltype(I),
                      decltype(S)> &&(I + 1
                                      < S) struct Increment<Pack::ValuesPack<I>,
                                                            Pack::ValuesPack<S>>
                  {
                    using type = Pack::ValuesPack<I + 1>;
                  };
    }  // namespace detail
    namespace detail
    {
    template<class T, auto S, auto... Ss>
    struct TensorTypeHelper
    {
      using type = Tensor<T, S, Ss...>;
    };

    template<class...>
    struct ContractHelper;
    template<auto N1, auto N2, auto... Ms, auto... Ps>
    struct ContractHelper<Pack::ValuesPack<N1, Ms...>,
                          Pack::ValuesPack<N2, Ps...>>
    {
      using ReverseM
          = ContractHelper<typename Pack::ValuesPack<N1, Ms...>::Reverse,
                           Pack::ValuesPack<N2, Ps...>>;
      using ReorderM
          = ContractHelper<typename Pack::ValuesPackConcat<
                               Pack::ValuesPack<N1>,
                               typename Pack::ValuesPack<Ms...>::Reverse>::type,
                           Pack::ValuesPack<N2, Ps...>>;
      static constexpr auto N = N1;
      static auto           Call(const auto& lhs, const auto& rhs)
      {
        return ReverseM::ReorderM::OrderedCall(lhs, rhs);
      }
      static auto OrderedCall(const auto& lhs,
                              const auto& rhs) requires(N1 == N2)
      {
        auto ret = typename detail::TensorTypeHelper<
            typename Traits::ElementAtDepth<std::remove_cvref_t<decltype(rhs)>,
                                            sizeof...(Ms)>::type,
            Ms...,
            Ps...>::type{};
        For<Ms...>::Call(
            [&](auto... ms)
            {
              For<Ps...>::Call(
                  [&](auto... ps)
                  {
                    For<N>::Call(
                        [&](auto n)
                        {
                          ret.at(ms..., ps...)
                              += lhs.at(ms..., n) * rhs.at(n, ps...);
                        });
                  });
            });
        return std::move(ret);
      }
    };
    template<auto N>
    struct ContractHelper<Pack::ValuesPack<N>, Pack::ValuesPack<N>>
    {
      static auto Call(const auto& lhs, const auto& rhs)
      {
        return OrderedCall(lhs, rhs);
      }
      static auto OrderedCall(const auto& lhs, const auto& rhs)
      {
        auto ret = typename Traits::Element<
            std::remove_cvref_t<decltype(lhs)>>::type{};
        For<N>::Call(
            [&](auto n)
            {
              ret += lhs.at(n) * rhs.at(n);
            });
        return std::move(ret);
      }
    };
    }  // namespace detail
    auto Contract(const Concepts::Tensor auto& lhs,
                  const Concepts::Tensor auto& rhs)
    {
      return detail::ContractHelper<
          Traits::SizePack<std::remove_cvref_t<decltype(lhs)>>,
          Traits::SizePack<std::remove_cvref_t<decltype(rhs)>>>::Call(lhs, rhs);
    }

    namespace detail
    {
    template<class... Ts>
    struct TensorProductHelper;
    template<auto V1, auto... V1s, auto V2, auto... V2s>
    struct TensorProductHelper<Pack::ValuesPack<V1, V1s...>,
                               Pack::ValuesPack<V2, V2s...>>
    {
      static auto Call(const auto& lhs, const auto& rhs)
      {
        auto ret = Tensor<typename Traits::BaseElement<
                              std::remove_cvref_t<decltype(lhs)>>::type,
                          V1,
                          V1s...,
                          V2,
                          V2s...>{};
        For<V1, V1s...>::Call(
            [&ret, &lhs, &rhs] (auto&& v1, auto&&... v1s) requires (sizeof...(v1s)==sizeof...(V1s)) //CLANG-FORMAT ERROR
            {
              For<V2, V2s...>::Call(
                  [&ret, &lhs, &rhs, &v1, &v1s...](auto&& v2, auto&&... v2s) requires (sizeof...(v2s)==sizeof...(V2s)) //CLANG-FORMAT ERROR
                  {
                    ret.at(v1, v1s..., v2, v2s...)
                        = lhs.at(
                              std::forward<std::remove_cvref_t<decltype(v1)>>(
                                  v1),
                              std::forward<std::remove_cvref_t<decltype(v1s)>>(
                                  v1s)...)
                        * rhs.at(
                            std::forward<std::remove_cvref_t<decltype(v2)>>(v2),
                            std::forward<std::remove_cvref_t<decltype(v2s)>>(
                                v2s)...);
                  });
            });
        return ret;
      }
    };
    }  // namespace detail
    auto TensorProduct(Concepts::Tensor auto&& lhs, Concepts::Tensor auto&& rhs)
    {
      return detail::TensorProductHelper<
          Traits::SizePack<std::remove_cvref_t<decltype(lhs)>>,
          Traits::SizePack<std::remove_cvref_t<decltype(rhs)>>>::
          Call(std::forward<decltype(lhs)>(lhs),
               std::forward<decltype(rhs)>(rhs));
    }
    auto ExteriorProduct(Concepts::Tensor auto&& lhs,
                         Concepts::Tensor auto&& rhs) requires Concepts::IdenticalRelative<std::remove_cvref_t<decltype(lhs)>, std::remove_cvref_t<decltype(rhs)>>
    {
      return TensorProduct(lhs, rhs)
           - TensorProduct(
                 std::forward<std::remove_cvref_t<decltype(rhs)>>(rhs),
                 std::forward<std::remove_cvref_t<decltype(lhs)>>(lhs));
    }
    auto CrossProduct(
        Concepts::Vector auto&& lhs,
        Concepts::Vector auto&&
            rhs) requires((Traits::Size<std::remove_cvref_t<decltype(lhs)>>::
                               value
                           == 3)
                          && (Traits::Size<
                                  std::remove_cvref_t<decltype(rhs)>>::value
                              == 3)
                          && (Traits::Rank<
                                  std::remove_cvref_t<decltype(lhs)>>::value
                              == 1)
                          && (Traits::Rank<
                                  std::remove_cvref_t<decltype(rhs)>>::value
                              == 1))
    {
      auto ext = ExteriorProduct(
          std::forward<std::remove_cvref_t<decltype(lhs)>>(lhs),
          std::forward<std::remove_cvref_t<decltype(rhs)>>(rhs));

      return Vector<typename Traits::BaseElement<
                        std::remove_cvref_t<decltype(lhs)>>::type,
                    3>{std::move(ext.at(1, 2)),
                       std::move(ext.at(2, 0)),
                       std::move(ext.at(0, 1))};
    }
    auto DotProduct(
        Concepts::Vector auto&& lhs,
        Concepts::Vector auto&&
            rhs) requires((Traits::Rank<std::remove_cvref_t<decltype(lhs)>>::
                               value
                           == 1)
                          && (Traits::Rank<
                                  std::remove_cvref_t<decltype(rhs)>>::value
                              == 1))
    {
      return Contract(std::forward<std::remove_cvref_t<decltype(lhs)>>(lhs),
                      std::forward<std::remove_cvref_t<decltype(rhs)>>(rhs));
    }
     auto InnerProduct(Concepts::Tensor auto&& lhs, Concepts::Tensor auto&& rhs)
     {
       return Trace(TensorProduct(std::forward<decltype(lhs)>(lhs),std::forward<decltype(rhs)>(rhs)));
     }
    auto Sum(const Concepts::Tensor auto& tensor)
    {
        auto sum = typename Traits::BaseElement<std::remove_cvref_t<decltype(tensor)>>::type{};
        for (auto v : tensor)
            sum += Sum(v);
        return sum;
    }
    auto Sum(const Concepts::BaseTensor auto& tensor)
    {
        auto sum = typename Traits::BaseElement<std::remove_cvref_t<decltype(tensor)>>::type{};
        for (auto v : tensor)
            sum += v;
        return sum;
    }
    namespace detail {
        auto MagnitudeHelper(const Concepts::Tensor auto& tensor){
            auto ret = typename Traits::BaseElement<std::remove_cvref_t<decltype(tensor)>>::type{};
            for (auto v : tensor)
                ret += MagnitudeHelper(v);
            return ret;
        }
        auto MagnitudeHelper(const Concepts::BaseTensor auto& tensor){
            auto ret = typename Traits::BaseElement<std::remove_cvref_t<decltype(tensor)>>::type{};
            for (auto v : tensor)
                ret += std::pow(v, 2);
            return ret;
        }
    }
    auto Magnitude(const Concepts::Tensor auto& tensor)
    {
        return std::sqrt(detail::MagnitudeHelper(tensor));
    }
    auto Normalize(const Concepts::Tensor auto& tensor)
    {
        return tensor / Magnitude(tensor);
    }
     //template<class T = typename Traits::Self>
     //  requires Concepts::Matrix<T>
     //auto Minors(const Concepts::Tensor auto& tensor)
     //{
     //  static constexpr tensor_size = Traits::Size<std::remove_cvref_t<decltype(tensor)>>::value;
     //  std::array<std::array<typename T::Traits::template Relative<-1>,
     //                        tensor_size>
     //      ret;
     //  for (Math::Type::ULong i = 0; i < tensor_size; ++i)
     //    ret[i] = Minor(i);
     //  return ret;
     //}
     //template<class T = typename Traits::Self>
     //  requires Concepts::Matrix<T> std::array <
     //      typename T::Traits::template Relative<-1>,
     //      T::Traits::Element::Traits::Size()
     //> Minors(const Math::Type::ULong& index)
     //{
     //  std::array<typename T::Traits::template Relative<-1>,
     //             T::Traits::Element::Traits::Size()>
     //      ret;
     //  for (Math::Type::ULong i = 0; i < T::Traits::Element::Traits::Size();
     //  ++i)
     //    ret[i] = Minor(index, i);
     //  return ret;
     //}
   //  template<class T = typename Traits::Self>
   //    requires Concepts::SquareMatrix<T>
   //  typename T::Traits::Type Determinant()
   //  {
   //    typename T::Traits::Type ret{};
   //    std::array<typename T::Traits::template Relative<-1>,
   //    T::Traits::Size()>
   //        minors{Minors(0)};
   //    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
   //      ret += (minors[i]) * (*this)[0][i] * (i % 2 == 1 ? -1 :
   //      1).Determinant();
   //    return ret;
   //  }
   //  template<class T = typename Traits::Self>
   //    requires(Concepts::SquareMatrix<T> && (T::Traits::Size() == 2))
   //  typename T::Traits::Type Determinant()
   //  {
   //    return (*this)[0][0] * (*this)[1][1] - (*this)[0][1] * (*this)[1][0];
   //  }
   //  template<class T = typename Traits::Self>
   //    requires(Concepts::SquareMatrix<T> && (T::Traits::Size() == 1))
   //  typename Traits::Type Determinant()
   //  {
   //    return (*this)[0][0];
   //  }
   //  template<class T = typename Traits::Self>
   //    requires Concepts::Matrix<T> &&(T::Traits::Size()
   //                                    == T::Traits::Element::Traits::Size()
   //                                    - 1)
   //        Vector<T::Traits::Element::Traits::Size(),
   //               typename T::Traits::Type> OrthogonalVector()
   //    {
   //      std::array<
   //          typename Matrix<T::Traits::Element::Traits::Size(),
   //                          T::Traits::Element::Traits::Size(),
   //                          T::Traits::Size()>::Traits::template
   //                          Relative<-1>,
   //          T::Traits::Element::Traits::Size()>
   //          arr{Minors(static_cast<Matrix<T::Traits::Element::Traits::Size(),
   //                                        T::Traits::Element::Traits::Size(),
   //                                        T::Traits::Size()>>(*this),
   //                     T::Traits::Element::Traits::Size() - 1)};
   //      Vector<T::Traits::Element::Traits::Size(), typename T::Traits::Type>
   //      ret; for (Math::Type::ULong i = 0; i <
   //      T::Traits::Element::Traits::Size(); ++i)
   //      {
   //        ret[i] = Determinant(arr[i]);
   //        if (T::Traits::Element::Traits::Size() % 2 == 0)
   //        {
   //          ret[i] *= (i % 2 == 1 ? 1 : -1);
   //        }
   //      }
   //      return ret;
   //    }
}  // namespace UD::Tensor
