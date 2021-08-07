#pragma once
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <iostream>
#include <algorithm>

#include <libudph/Math/Math.h>
#include <libudph/Class/Traits.h>
#include <libudph/Class/SFINAE.h>
#include <libudph/Container/Container.h>

namespace UD
{
namespace Tensor
{
template<Math::Type::ULong _Size,
         class _Element,
         class _Type = typename _Element::Traits::Type>
struct Tensor;
template<class _Type, Math::Type::ULong _Size, Math::Type::ULong... _Sizes>
struct PureTensorHelper
{
  using Type = Tensor<_Size, PureTensorHelper<_Type, _Sizes...>, _Type>;
};
template<class _Type, Math::Type::ULong _Size>
struct PureTensorHelper<_Type, _Size>
{
  using Type = _Type;
};
template<class Type, Math::Type::ULong Size, Math::Type::ULong... Sizes>
using PureTensor
    = Tensor<Size, typename PureTensorHelper<Type, Size, Sizes...>::Type, Type>;
template<Math::Type::ULong N, class Element, class Type = Element>
using Vector = Tensor<N, Element, Type>;
template<class Type, std::size_t N>
using PureVector = Vector<N, Type>;
template<Math::Type::ULong N,
         Math::Type::ULong M,
         class Element,
         class Type = Element>
using Matrix = Tensor<M, Vector<N, Element, Type>>;
template<class Type, std::size_t N, std::size_t M>
using PureMatrix = Matrix<N, M, Type>;
}  // namespace Tensor
namespace Traits
{
namespace detail
{
template<class T>
struct TensorBase;
template<Math::Type::ULong _Size, class _Element, class _Type>
struct TensorBase<Tensor::Tensor<_Size, _Element, _Type>>
{
  using Type = _Type;
};
}  // namespace detail
template<Math::Type::ULong _Size, class _Element, class _Type, class _Leaf>
  requires(!std::is_same_v<_Element, _Type>)
struct Register<Tensor::Tensor<_Size, _Element, _Type>, _Leaf>
    : public detail::TensorBase<Tensor::Tensor<_Size, _Element, _Type>>
    , public Inherit<_Leaf, Container::Array<_Element, _Size>>
{
  static constexpr Math::Type::ULong Order()
  {
    return 1 + Register::Element::Traits::Order();
  }

 private:
  template<Math::Type::Long Adjustment>
  struct RelativeHelper
      : public Tensor::Tensor<
            Math::Clamp<Math::Type::ULong, Math::Type::ULong>(
                Math::Clamp<Math::Type::ULong, Math::Type::Long>(
                    Register::Size())
                + Adjustment),
            typename Register::Element::Traits::template Relative<Adjustment>,
            typename Register::Type>
  {
  };

 public:
  template<Math::Type::Long Adjustment>
  using Relative = RelativeHelper<Adjustment>;
};
template<Math::Type::ULong _Size, class _Type, class _Leaf>
struct Register<Tensor::Tensor<_Size, _Type, _Type>, _Leaf>
    : public detail::TensorBase<Tensor::Tensor<_Size, _Type, _Type>>
    , public Inherit<_Leaf, Container::Array<_Type, _Size>>
{
  static constexpr Math::Type::ULong Order()
  {
    return 1;
  }
  template<Math::Type::Long Adjustment>
  using Relative
      = Tensor::Tensor<Math::Clamp<Math::Type::Long, Math::Type::ULong>(
                           Math::Clamp<Math::Type::ULong, Math::Type::Long>(
                               Register::Size())
                           + Adjustment),
                       typename Register::Element,
                       typename Register::Type>;
};
// template <class T, class _Derived>
// struct SimpleTensorTraits;
// template <Math::Type::ULong _Size, class _Element, class _Type, class
// _Derived> struct SimpleTensorTraits<Tensor::Tensor<_Size, _Element, _Type>,
// _Derived>
//    : public Traits<Container::Array<_Element, _Size>, _Derived> {
// private:
//  using Base = Traits<Container::Array<_Element, _Size>, _Derived>;
//
// public:
//  using Type = _Type;
//};
// template <Math::Type::ULong _Size, class _Element, class _Type, class
// _Derived> requires(!std::is_same_v<_Element, _Type>) struct Traits<
//    Tensor::Tensor<_Size, _Element, _Type>, _Derived>
//    : public SimpleTensorTraits<Tensor::Tensor<_Size, _Element, _Type>,
//                                _Derived> {
// private:
//  using Base =
//      SimpleTensorTraits<Tensor::Tensor<_Size, _Element, _Type>, _Derived>;
//
// public:
//  static constexpr Math::Type::ULong Order() {
//    return 1 + _Element::Traits::Order();
//  }
//
// private:
//  template <Math::Type::Long Adjustment>
//  struct RelativeHelper
//      : public Tensor::Tensor<
//            Math::Clamp<Math::Type::ULong, Math::Type::ULong>(
//                Math::Clamp<Math::Type::ULong, Math::Type::Long>(Base::Size())
//                + Adjustment),
//            typename Traits<typename Traits::Element>::template
//            RelativeHelper<
//                Adjustment>,
//            typename Base::Type> {};
//
// public:
//  template <Math::Type::Long Adjustment>
//  using Relative = RelativeHelper<Adjustment>;
//};
// template <Math::Type::ULong _Size, class _Type, class _Derived>
// struct Traits<Tensor::Tensor<_Size, _Type, _Type>, _Derived>
//    : public SimpleTensorTraits<Tensor::Tensor<_Size, _Type, _Type>, _Derived>
//    {
// private:
//  using Base =
//      SimpleTensorTraits<Tensor::Tensor<_Size, _Type, _Type>, _Derived>;
//
// public:
//  static constexpr Math::Type::ULong Order() { return 1; }
//  template <Math::Type::Long Adjustment>
//  using Relative =
//      Tensor::Tensor<Math::Clamp<Math::Type::Long, Math::Type::ULong>(
//                         Math::Clamp<Math::Type::ULong, Math::Type::Long>(
//                             Base::Size()) +
//                         Adjustment),
//                     typename Base::Element, typename Base::Type>;
//};
}  // namespace Traits
namespace Tensor::Concepts
{
template<class T>
concept False = false;
template<class T>
concept Tensor = requires(T t)
{
  typename Traits::Traits<T>;
  typename Traits::Traits<T>::Type;
  Math::Concepts::Arithmeticable<typename Traits::Traits<T>::Type>;
  typename Traits::Traits<T>::Element;
  std::same_as<typename Traits::Traits<T>::Element,
               typename Traits::Traits<T>::Type> || requires()
  {
    Tensor<typename Traits::Traits<T>::Element>;
    std::same_as<
        typename Traits::Traits<typename Traits::Traits<T>::Element>::Type,
        typename Traits::Traits<T>::Type>;
  };
  {
    Traits::Traits<T>::Size()
    } -> std::convertible_to<const Math::Type::ULong>;
  {
    Traits::Traits<T>::Order()
    } -> std::convertible_to<const Math::Type::ULong>;
};
template<class T>
concept BaseTensor
    = Tensor<T> && std::same_as < typename Traits::Traits<T>::Element,
typename Traits::Traits<T>::Type > ;
template<class T>
concept Vector = BaseTensor<T>;
template<class T>
concept Matrix = Tensor<T> && Vector<typename Traits::Traits<T>::Element>;
template<class T>
concept SquareMatrix = Matrix<T> &&(
    Traits::Traits<T>::Size
    == Traits::Traits<typename Traits::Traits<T>::Element>::Size);
template<class T, class U, Math::Type::Long Size, Math::Type::Long... Sizes>
concept Relative
    = Tensor<T> && Tensor<U> && std::same_as < typename Traits::Traits<T>::Type,
typename Traits::Traits<U>::Type > &&Traits::Traits<T>::Order
        == Traits::Traits<U>::Order
    && (Traits::Traits<T>::Size + Size == Traits::Traits<U>::Size)
    && (BaseTensor<
            T> || Relative<typename Traits::Traits<T>::Element, typename Traits::Traits<U>::Element, Sizes...>);
template<class T, class U>
concept SubTensor
    = Tensor<T> && Tensor<U> && std::same_as < typename Traits::Traits<T>::Type,
typename Traits::Traits<U>::Type > &&(
    (Traits::Traits<T>::Order == Traits::Traits<U>::Order
     && Traits::Traits<T>::Size <= Traits::Traits<U>::Size
     && (BaseTensor<
             T> || SubTensor<typename Traits::Traits<T>::Element, typename Traits::Traits<U>::Element>))
    || (Traits::Traits<T>::Order < Traits::Traits<U>::Order
        && SubTensor<T, typename Traits::Traits<U>::Element>));
template<class T, class U>
concept SuperTensor = SubTensor<U, T>;
template<class T>
concept InterfacedTensor = Tensor<T> && requires(T t)
{
  requires requires(Math::Type::ULong i)
  {
    {
      t[i]
      } -> std::same_as<typename Traits::Traits<T>::Element>;
    {
      t.at(i)
      } -> std::same_as<typename Traits::Traits<T>::Element>;
  };
  requires requires(Math::Type::ULong first, Math::Type::ULong last)
  {
    requires last <= Traits::Traits<T>::Size&& first <= last;
    {
      t.Sub(first, last)
      } -> Relative<T, -(last - first)>;
  };
};
}  // namespace Tensor::Concepts
namespace Tensor
{
template<Math::Type::ULong _Size, class _Element, class _Type>
struct Tensor : public Container::Array<_Element, _Size>
{
  using Traits = Traits::Traits<Tensor>;
  static_assert(Concepts::template Tensor<Tensor>,
                "Does not meet Tensor concept requirements.");

 private:
  using Base = Container::Array<typename Traits::Element, Traits::Size()>;
  template<Math::Type::ULong SizeB, class ElementB, class TypeB>
    requires(
        Concepts::SubTensor<
            Tensor<SizeB, ElementB, TypeB>,
            typename Traits::
                Self> && (Traits::Order() == Tensor<SizeB, ElementB, TypeB>::Traits::Order()))
  static Tensor Construct(const Tensor<SizeB, ElementB, TypeB>& other)
  {
    Tensor ret;
    for (Math::Type::ULong i = 0; i < SizeB; ++i)
      ret[i] = other[i];
    return ret;
  }
  template<Math::Type::ULong SizeB, class ElementB, class TypeB>
    requires(Concepts::SubTensor<Tensor<SizeB, ElementB, TypeB>,
                                 typename Traits::Self>&& Traits::Order()
             > Tensor<SizeB, ElementB, TypeB>::Traits::Order())
  static Tensor Construct(const Tensor<SizeB, ElementB, TypeB>& other)
  {
    Tensor ret;
    *ret.front() = typename Traits::Element::Construct(other);
    return ret;
  }

 public:
  virtual ~Tensor()         = default;
  Tensor(const Tensor&)     = default;
  Tensor(Tensor&&) noexcept = default;
  auto operator=(const Tensor&) -> Tensor& = default;
  auto operator=(Tensor&&) noexcept -> Tensor& = default;
  using Base::Base;
  // template <Math::Type::ULong SizeB, class ElementB, class TypeB>
  // requires Concepts::SubTensor<Tensor<SizeB, ElementB, TypeB>,
  //                             typename Traits::Self>
  // Tensor(const Tensor<SizeB, ElementB, TypeB>& other) {
  //  *this = Construct(other);
  //}
  // Tensor() {}
  // template <class... Ts>
  // Tensor(Ts... ts) : Base{ts...} {}
  // template <class... Ts>
  // requires requires() {
  //  std::conjunction_v<std::is_convertible_v<Ts, typename
  //  Traits::Element>...>; requires sizeof...(Ts) == Traits::Size();
  //}
  // Tensor(Ts... ts) : Base{ts...} {}
  // Tensor(const Tensor&) = default;
  // Tensor(Tensor&&) = default;
  // Tensor& operator=(const Tensor&) = default;
  // Tensor& operator=(Tensor&&) = default;
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 1)
  typename Traits::Element& x()
  {
    return this->at(0);
  }
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 2)
  typename Traits::Element& y()
  {
    return this->at(1);
  }
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 3)
  typename Traits::Element& z()
  {
    return this->at(2);
  }
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 1)
  const typename Traits::Element& x() const
  {
    return this->at(0);
  }
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 2)
  const typename Traits::Element& y() const
  {
    return this->at(1);
  }
  template<Math::Type::ULong S = Traits::Size()>
    requires(S >= 3)
  const typename Traits::Element& z() const
  {
    return this->at(2);
  }
  Tensor& operator+=(const typename Traits::Type& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) += rhs;
    return *this;
  }
  Tensor& operator+=(const Tensor& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) += rhs[i];
    return *this;
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator+(T lhs, const typename Traits::Type& rhs)
  {
    return static_cast<T&>(lhs += rhs);
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator+(T lhs, const Tensor& rhs)
  {
    return static_cast<T&>(lhs += rhs);
  }
  Tensor& operator-=(const typename Traits::Type& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) -= rhs;
    return *this;
  }
  Tensor& operator-=(const Tensor& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) -= rhs[i];
    return *this;
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator-(T lhs, const typename Traits::Type& rhs)
  {
    return static_cast<T&>(lhs -= rhs);
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator-(T lhs, const Tensor& rhs)
  {
    return static_cast<T&>(lhs -= rhs);
  }
  Tensor& operator*=(const typename Traits::Type& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) *= rhs;
    return *this;
  }
  Tensor& operator*=(const Tensor& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) *= rhs[i];
    return *this;
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator*(T lhs, const typename Traits::Type& rhs)
  {
    return static_cast<T&>(lhs *= rhs);
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator*(T lhs, const Tensor& rhs)
  {
    return static_cast<T&>(lhs *= rhs);
  }
  Tensor& operator/=(const typename Traits::Type& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) /= rhs;
    return *this;
  }
  Tensor& operator/=(const Tensor& rhs)
  {
    for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
      this->at(i) /= rhs[i];
    return *this;
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator/(T lhs, const typename Traits::Type& rhs)
  {
    return static_cast<T&>(lhs /= rhs);
  }
  template<class T = Tensor>
    requires std::derived_from<T, Tensor>
  friend T operator/(T lhs, const Tensor& rhs)
  {
    return static_cast<T&>(lhs /= rhs);
  }
  Tensor operator-() const
  {
    return *this * static_cast<typename Traits::Type>(-1);
  }
  friend std::ostream& operator<<(std::ostream& os, const Tensor& a)
  {
    if (Traits::Size() == 0)
      return os << "[]";
    os << '[';
    for (Math::Type::ULong i = 0; i < Traits::Size() - 1; ++i)
      os << a[i] << ',';
    return os << a[Traits::Size() - 1] << ']';
  }
  // Merge case: lhs convertible to rhs || not rhs convertible to lhs
  template<Math::Type::ULong Size2, class Element2, class Type2>
    requires(
        std::is_convertible_v<
            Tensor,
            Tensor<
                Size2,
                Element2,
                Type2>> || !std::is_convertible_v<Tensor<Size2, Element2, Type2>, Tensor>)
  Tensor<Traits::Size() + Size2, Element2, Type2> Merge(
      const Tensor<Size2, Element2, Type2>& rhs)
  {
    Tensor<Traits::Size() + Size2, Element2, Type2> ret{};
    ret = *this;
    for (Math::Type::ULong i = 0; i < Size2; ++i)
      ret[Traits::Size() + i] = rhs[i];
    return ret;
  }
  // Merge case: not lhs convertible to rhs && rhs convertible to lhs
  template<Math::Type::ULong Size2, class Element2, class Type2>
    requires(
        !std::is_convertible_v<
            Tensor,
            Tensor<
                Size2,
                Element2,
                Type2>> && std::is_convertible_v<Tensor<Size2, Element2, Type2>, Tensor>)
  Tensor<Traits::Size() + Size2,
         typename Traits::Element,
         typename Traits::Type>
      Merge(const Tensor<Traits::Size(),
                         typename Traits::Element,
                         typename Traits::Type>&  lhs,
            const Tensor<Size2, Element2, Type2>& rhs)
  {
    Tensor<Traits::Size() + Size2,
           typename Traits::Element,
           typename Traits::Type>
        ret{};
    ret = *this;
    for (Math::Type::ULong i = 0; i < Size2; ++i)
      ret[Traits::Size() + i] = rhs[i];
    return ret;
  }
  typename Traits::template Relative<-1> Minor(const Math::Type::ULong& index)
  {
    typename Traits::template Relative<-1> ret{};
    for (Math::Type::ULong i = 0; i < index; ++i)
      ret[i] = this->at(i);
    for (Math::Type::ULong i = index + 1; i < Traits::Size(); ++i)
      ret[i - 1] = this->at(i);
    return ret;
  }
  template<class... Indices>
    requires(
        std::conjunction_v<
            std::is_same<Math::Type::ULong,
                         Indices>...> && sizeof...(Indices) == Traits::Order())
  typename Traits::template Relative<-1> Minor(const Math::Type::ULong& index,
                                               Indices... indices)
  {
    typename Traits::template Relative<-1> ret{};
    for (Math::Type::ULong i = 0; i < index; ++i)
      ret[i] = Minor(this->at(i), indices...);
    for (Math::Type::ULong i = index + 1; i < Traits::Size(); ++i)
      ret[i - 1] = Minor(this->at(i), indices...);
    return ret;
  }
  template<class T = typename Traits::Self>
    requires Concepts::Matrix<T> Matrix<T::Traits::Element::Traits::Size(),
                                        Traits::Size(),
                                        typename T::Traits::Type>
                                 Transpose()
    {
      Matrix<T::Traits::Element::Traits::Size(),
             T::Traits::Size(),
             typename T::Traits::Type>
          ret;
      for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
      {
        for (Math::Type::ULong j = 0; j < T::Traits::Element::Traits::Size();
             ++j)
        {
          ret[j][i] = (*this)[i][j];
        }
      }
      return ret;
    }
  template<class T = typename Traits::Self>
    requires Concepts::SquareMatrix<T>
  typename T::Traits::Type Trace()
  {
    typename T::Traits::Type ret{0};
    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
    {
      ret += this->at(i).at(i);
    }
    return ret;
  }
  template<class T = typename Traits::Self>
    requires Concepts::SquareMatrix<T>
  typename T::Traits::Type RevTrace()
  {
    typename T::Traits::Type ret{0};
    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
    {
      ret += (*this)[i][T::Traits::Size() - i - 1];
    }
    return ret;
  }
  template<Math::Type::ULong S, class T = typename Traits::Self>
    requires Concepts::SquareMatrix<T>
        Matrix<T::Traits::Size(), S, typename T::Traits::Type>
  operator*(const Matrix<T::Traits::Element::Traits::Size(),
                         S,
                         typename T::Traits::Type>& rhs)
  {
    Matrix<T::Traits::Size(), S, typename T::Traits::Type> ret{};
    Matrix<S, T::Traits::Element::Traits::Size(), typename T::Traits::Type>
        trhs{Transpose(rhs)};
    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
    {
      for (Math::Type::ULong j = 0; j < S; ++j)
      {
        ret[i][j] = this->DotProduct(trhs[j]);
      }
    }
    return ret;
  }
  // Generalized Outer Product
  template<Math::Type::ULong S, class T = typename Traits::Self>
    requires Concepts::Vector<T>
        Matrix<T::Traits::Size(), S, typename T::Traits::Type> TensorProduct(
            const Vector<S, typename T::Traits::Type>& rhs)
    {
      return Transpose() * Matrix<1, S, typename T::Traits::Type>(rhs);
    }
  // Exterior/Wedge Product - generalized Cross Product
  template<Math::Type::ULong M, class T = typename Traits::Self>
    requires Concepts::Vector<T>
        Matrix<T::Traits::Size(), M, typename T::Traits::Type> ExteriorProduct(
            const Vector<M, typename T::Traits::Type>& rhs)
    {
      return TensorProduct(rhs) - rhs.TensorProduct(*this);
    }
  // Inner Product - generalized Dot Product
  template<class T = typename Traits::Self>
    requires Concepts::Vector<T>
  typename T::Traits::Type InnerProduct(
      const Vector<T::Traits::Size(), typename T::Traits::Type>& rhs)
  {
    return Trace(TensorProduct(rhs));
  }
  template<class T = typename Traits::Self>
    requires Concepts::SquareMatrix<T>
  typename T::Traits::Type Determinant()
  {
    typename T::Traits::Type ret{};
    std::array<typename T::Traits::template Relative<-1>, T::Traits::Size()>
        minors{Minors(0)};
    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
      ret += (minors[i]) * (*this)[0][i] * (i % 2 == 1 ? -1 : 1).Determinant();
    return ret;
  }
  template<class T = typename Traits::Self>
    requires(Concepts::SquareMatrix<T> && (T::Traits::Size() == 2))
  typename T::Traits::Type Determinant()
  {
    return (*this)[0][0] * (*this)[1][1] - (*this)[0][1] * (*this)[1][0];
  }
  template<class T = typename Traits::Self>
    requires(Concepts::SquareMatrix<T> && (T::Traits::Size() == 1))
  typename Traits::Type Determinant()
  {
    return (*this)[0][0];
  }
  template<class T = typename Traits::Self>
    requires Concepts::Matrix<T> std::array < std::array <
        typename T::Traits::template Relative<-1>,
        T::Traits::Element::Traits::Size()
  >, T::Traits::Size() > Minors()
  {
    std::array<std::array<typename T::Traits::template Relative<-1>,
                          T::Traits::Element::Traits::Size()>,
               T::Traits::Size()>
        ret;
    for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
      ret[i] = Minors(i);
    return ret;
  }
  template<class T = typename Traits::Self>
    requires Concepts::Matrix<T> std::array <
        typename T::Traits::template Relative<-1>,
        T::Traits::Element::Traits::Size()
  > Minors(const Math::Type::ULong& index)
  {
    std::array<typename T::Traits::template Relative<-1>,
               T::Traits::Element::Traits::Size()>
        ret;
    for (Math::Type::ULong i = 0; i < T::Traits::Element::Traits::Size(); ++i)
      ret[i] = Minor(index, i);
    return ret;
  }
  template<class T = typename Traits::Self>
    requires Concepts::Matrix<T> &&(T::Traits::Size()
                                    == T::Traits::Element::Traits::Size() - 1)
        Vector<T::Traits::Element::Traits::Size(),
               typename T::Traits::Type> OrthogonalVector()
    {
      std::array<
          typename Matrix<T::Traits::Element::Traits::Size(),
                          T::Traits::Element::Traits::Size(),
                          T::Traits::Size()>::Traits::template Relative<-1>,
          T::Traits::Element::Traits::Size()>
          arr{Minors(static_cast<Matrix<T::Traits::Element::Traits::Size(),
                                        T::Traits::Element::Traits::Size(),
                                        T::Traits::Size()>>(*this),
                     T::Traits::Element::Traits::Size() - 1)};
      Vector<T::Traits::Element::Traits::Size(), typename T::Traits::Type> ret;
      for (Math::Type::ULong i = 0; i < T::Traits::Element::Traits::Size(); ++i)
      {
        ret[i] = Determinant(arr[i]);
        if (T::Traits::Element::Traits::Size() % 2 == 0)
        {
          ret[i] *= (i % 2 == 1 ? 1 : -1);
        }
      }
      return ret;
    }
    // Vector
    template<class T = typename Traits::Self>
      requires Concepts::Vector<T>
    typename T::Traits::Type DotProduct(const Tensor& rhs)
    {
      typename T::Traits::Type ret{0};
      for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
      {
        ret += this->at(i) * rhs[i];
      }
      return ret;
    }
    template<class T = typename Traits::Self>
      requires Concepts::Vector<T>
          Matrix<T::Traits::Size(), 1, typename T::Traits::Type> Transpose()
      {
        return Transpose(
            Matrix<1, T::Traits::Size(), typename T::Traits::Type>(*this));
      }
    template<class T = typename Traits::Self>
      requires Concepts::Vector<T> Tensor Normalize()
      {
        typename T::Traits::Type sum{0};
        for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
        {
          sum += std::pow(this->at(i), 2);
        }
        return *this / std::sqrt(sum);
      }
    template<class T = typename Traits::Self>
      requires(Concepts::Vector<T>&& T::Traits::Size() == 3)
    Tensor CrossProduct(const Tensor& rhs)
    {
      Matrix<T::Traits::Size(), T::Traits::Size(), typename T::Traits::Type> m{
          this->ExteriorProduct(rhs)};
      return Tensor{m[1][2], m[2][0], m[0][1]};
    }
    void Scale(const typename Traits::Type& rhs)
    {
      *this *= rhs;
    }
    void Scale(const Tensor& rhs)
    {
      for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
        this->at(i) *= rhs[i];
    }
    template<Math::Type::ULong Size2, class Element2, class Type2>
      requires(Concepts::SubTensor<Tensor<Size2, Element2, Type2>,
                                   typename Traits::Self>)
    void Scale(const Tensor<Size2, Element2, Type2>& rhs)
    {
      for (auto& it : *this)
        Scale(it, rhs);
    }
    void Translate(const typename Traits::Type& rhs)
    {
      *this += rhs;
    }
    void Translate(const Tensor& rhs)
    {
      for (Math::Type::ULong i = 0; i < Traits::Size(); ++i)
        this->at(i) += rhs[i];
    }
    template<Math::Type::ULong Size2, class Element2, class Type2>
      requires(Concepts::SubTensor<Tensor<Size2, Element2, Type2>,
                                   typename Traits::Self>)
    void Translate(const Tensor<Size2, Element2, Type2>& rhs)
    {
      for (auto& it : *this)
        Translate(it, rhs);
    }
    template<class T = typename Traits::Self>
      requires Concepts::Vector<T>
    typename T::Traits::Type Magnitude()
    {
      typename T::Traits::Type sum = 0;
      for (auto& v : *this)
        sum += std::pow(v, 2);
      return std::sqrt(sum);
    }
    template<class T = typename Traits::Self>
      requires Concepts::Vector<T> Tensor UnitVector()
      {
        return *this / this->Magnitude();
      }
    typename Traits::Element Sum() const
    {
      typename Traits::Element sum = 0;
      for (const auto& v : *this)
        sum += v;
      return sum;
    }
    template<class T = typename Traits::Self>
      requires !Concepts::Vector<T> Tensor Transform(
          const Tensor&                         rhs,
          std::function<typename T::Traits::Type(
              const typename T::Traits::Type&,
              const typename T::Traits::Type&)> op)
      {
        Tensor ret;
        for (Math::Type::ULong i = 0; i < T::Traits::Size(); ++i)
          ret[i] = this->Transform(rhs[i], op);
        return ret;
      }
      template<class T = typename Traits::Self>
        requires Concepts::Vector<T> Tensor Transform(
            const Tensor& rhs,
            std::function<
                typename T::Traits::Type(const typename T::Traits::Type&,
                                         const typename T::Traits::Type&)> op)
        {
          Tensor ret;
          std::transform(this->begin(),
                         this->end(),
                         rhs.begin(),
                         ret.begin(),
                         op);
          return ret;
        }
      Tensor Mutliply(const Tensor& rhs)
      {
        return this->Transform(rhs, std::multiplies);
      }
};
}  // namespace Tensor
}  // namespace UD
