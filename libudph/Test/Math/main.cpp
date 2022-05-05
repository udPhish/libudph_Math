#include <array>
#include <chrono>
#include <concepts>
#include <forward_list>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <tuple>
#include <type_traits>

#include <libudph/Math/Tensor.h>

struct testing
{
  template<std::size_t i, std::size_t... is>
    requires(sizeof...(is) > 0)
  int t()
  {
    std::cout << "t1" << std::endl;
    return 1;
  }
  template<std::size_t i>
  unsigned int t()
  {
    std::cout << "t2" << std::endl;
    return 2;
  }
  template<std::same_as<std::size_t> i, std::same_as<std::size_t>... is>
    requires(sizeof...(is) > 0)
  int t(i, is...)
  {
    std::cout << "t1" << std::endl;
    return 1;
  }
  template<std::same_as<std::size_t> i>
  unsigned int t(i)
  {
    std::cout << "t2" << std::endl;
    return 2;
  }
};
auto calc(UD::Concepts::Tensor::Tensor auto f)
{
  return f;
}
int sum(int f, int e)
{
  return f + e;
}
struct val
{
  val() = default;
  val(const int&) {}
  operator int() const
  {
    return 2;
  }
  operator long()
  {
    return 2;
  }
  friend val operator+(const val& lhs, const val& rhs)
  {
    return val{};
  }
  val& operator+=(const val& rhs)
  {
    return *this;
  }
  friend val operator-(const val& lhs, const val& rhs)
  {
    return val{};
  }
  val& operator-=(const val& rhs)
  {
    return *this;
  }
  friend val operator/(const val& lhs, const val& rhs)
  {
    return val{};
  }
  val& operator/=(const val& rhs)
  {
    return *this;
  }
  friend val operator*(const val& lhs, const val& rhs)
  {
    return val{};
  }
  val& operator*=(const val& rhs)
  {
    return *this;
  }
};
// template<UD::Concepts::Tensor::CSizedRange T, class Element>
//   requires UD::Concepts::Tensor::RangeOf<T, Element>
// void tp()
//{
//   std::cout << "RangeOf" << std::endl;
// }
// template<UD::Concepts::Tensor::CSizedRange T, class Element>
//   requires UD::Concepts::Tensor::SimpleRangeOf<T, Element>
// void tp()
//{
//   std::cout << "SimpleRangeOf" << std::endl;
// }
template<class T>
concept Any = true;
template<class T>
concept AnyElse = Any<T> && true;
template<class T>
struct dosome;
template<class T, UD::Math::Type::ULong S>
struct dosome<UD::Tensor::Matrix<T, S, S>>
{
};
template<class T>
void pvalpack()
{
  std::cout << T::Value << std::endl;
}
template<class T>
  requires(!T::Next::Empty)
void pvalpack()
{
  std::cout << T::Value << ",";
  pvalpack<typename T::Next>();
}
template<class T, class U>
void pp()
{
  pvalpack<T>();
}
template<class T, class U>
  requires requires()
  {
    typename UD::Tensor::detail::Increment<T, U>::type;
  } &&(!UD::Tensor::detail::TrailingZero<
       typename UD::Tensor::detail::Increment<T, U>::type>::value) void pp()
  {
    pvalpack<T>();
    pp<typename UD::Tensor::detail::Increment<T, U>::type, U>();
  }

  template<class Ts>
  struct test;

  template<int T, int... Ts>
  struct test<UD::Pack::ValuesPack<T, Ts...>>
  {
    static void Print() requires(sizeof...(Ts) > 0)
    {
      std::cout << T << ",";
      test<UD::Pack::ValuesPack<Ts...>>::Print();
    }
    static void Print() requires(sizeof...(Ts) == 0)
    {
      std::cout << T << std::endl;
    }
  };
  template<class T>
  void printem(T t)
  {
    std::cout << t << std::endl;
  }
  template<class T, class... Ts>
    requires(sizeof...(Ts) > 0)
  void printem(T t, Ts... ts)
  {
    std::cout << t << ",";
    printem(ts...);
  }
  template<auto V, auto... Vs>
  void count()
  {
    UD::For<V, Vs...>::Call(printem<decltype(V), decltype(Vs)...>);
  }
  template<auto V, auto... Vs>
  void countr()
  {
    UD::For<V, Vs...>::CallReverse(printem<decltype(V), decltype(Vs)...>);
  }

  int main(int argc, char** argv)
  {
    using namespace UD::Tensor;
    using Type1 = UD::Tensor::Tensor<int, 3, 3>;
    using Type2 = UD::Tensor::Tensor<int, 3, 2, 1>;

    using ArrType1 = std::array<std::array<int, 1>, 2>;

    auto t1 = Type1{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto t2 = Type1{9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::cout << IndexMultiplication<UD::Pack::ValuesPack<0>,
                                     UD::Pack::ValuesPack<2>>(t1, t2)
              << std::endl;

    pp<UD::Pack::ValuesPack<0, 0>, UD::Pack::ValuesPack<2, 2>>();
    typename detail::Increment<UD::Pack::ValuesPack<0, 0>,
                               UD::Pack::ValuesPack<2, 2>>::type item;

    pvalpack<UD::Pack::ValuesPack<1, 2, 3, 4, 5, 6, 6, 7, 8, 9>>();
    pvalpack<UD::Pack::ValuesPack<1, 2, 3, 4, 5, 6, 6, 7, 8, 9>::Reverse>();

    std::cout << Tensor<int, 2, 3, 4>{4, 2, 1, 3, 5, 3, 7, 5, 5, 7, 6, 5, 6}
              << std::endl;
    std::cout << Tensor<int, 2, 3, 4>{4, 2, 1, 3, 5, 3, 7, 5, 5, 7, 6, 5, 6}
                     .at<0, 1, 1>()
              << std::endl;
    std::cout
        << Tensor<int, 2, 3, 4>{4, 2, 1, 3, 5, 3, 7, 5, 5, 7, 6, 5, 6}.at<1>()
        << std::endl;
    std::cout << Tensor<int, 2, 3, 4>{4, 2, 1, 3, 5, 3, 7, 5, 5, 7, 6, 5, 6}
                     .rat(2, 1, 0)
              << std::endl;
    std::cout
        << Tensor<int, 2, 3, 4>{4, 2, 1, 3, 5, 3, 7, 5, 5, 7, 6, 5, 6}.rat(1)
        << std::endl;

    UD::Pack::Pack<int, int, int>::Values<1, 2, 3> ttttt;
    std::cout << UD::Pack::Pack<int, int, int>::Values<1, 2, 3>::Value
              << std::endl;
    std::cout << UD::Pack::Pack<int, int, int>::Values<1, 2, 3>::Next::Value
              << std::endl;
    std::cout
        << UD::Pack::Pack<int, int, int>::Values<1, 2, 3>::Next::Next::Value
        << std::endl;
    std::cout << UD::Pack::Pack<int, int, int>::Values<1, 2, 3>::Concat<
        UD::Pack::Pack<int, int, int>::Values<3, 4, 123>>::LastValue
              << std::endl;

    std::cout << Tensor<int, 2, 3>{4, 2, 1, 3, 5, 3} << std::endl;

    std::cout << UD::Tensor::Transpose<int, 2, 3>(
        Matrix<int, 2, 3>{4, 2, 1, 3, 5, 3})
              << std::endl;
    auto [a, b, c, d, e, f]
        = Tensor<int, 2, 3, 213, 5, 123, 2>{4, 2, 1, 3, 5, 3}.sizes();
    std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f
              << std::endl;

    std::cout << "Trace: "
              << Trace(Matrix<int, 3, 3>{1, 2, 3, 4, 3, 2, 12, 4, 5})
              << std::endl;

    std::cout << "Contraction:" << std::endl;
    auto contraction1 = Matrix<int, 2, 3>{1, 2, 3, 4, 5, 6};
    auto contraction2 = Matrix<int, 3, 2>{7, 8, 9, 10, 11, 12};
    std::cout << contraction1 << std::endl;
    std::cout << contraction2 << std::endl;
    std::cout << Contract(contraction1, contraction2) << std::endl;

    static_assert(UD::Concepts::Tensor::OfSize<Tensor<int, 2, 3, 4>, 2, 3, 4>);
    static_assert(
        std::same_as<
            typename UD::Traits::Tensor::ElementAtDepth<Tensor<int, 8, 3, 4>,
                                                        0>::type,
            typename UD::Traits::Tensor::ElementAtDepth<Tensor<int, 2, 3, 4>,
                                                        0>::type>);

    auto tprod1 = Tensor<double, 3>{1, 2, 3};
    auto tprod2 = Tensor<double, 3>{67, -2, 1};
    std::cout << tprod1 << std::endl;
    std::cout << tprod2 << std::endl;
    std::cout << "TensorProduct: " << std::endl;
    std::cout << TensorProduct(tprod1, tprod2) << std::endl;
    std::cout << "ExteriorProduct: " << std::endl;
    std::cout << ExteriorProduct(tprod1, tprod2) << std::endl;
    std::cout << "CrossProduct: " << std::endl;
    std::cout << CrossProduct(tprod1, tprod2) << std::endl;
    std::cout << "Contract: " << std::endl;
    std::cout << Contract(tprod1, tprod2) << std::endl;
    std::cout << "DotProduct: " << std::endl;
    std::cout << DotProduct(tprod1, tprod2) << std::endl;
    std::cout << "InnerProduct: " << std::endl;
    std::cout << InnerProduct(tprod1, tprod2) << std::endl;
    std::cout << "Sum: " << std::endl;
    std::cout << Sum(tprod1) << std::endl;
    std::cout << Sum(tprod2) << std::endl;
    std::cout << "Magnitude: " << std::endl;
    std::cout << Magnitude(tprod1) << std::endl;
    std::cout << Magnitude(tprod2) << std::endl;
    std::cout << "Normalize: " << std::endl;
    std::cout << Normalize(tprod1) << std::endl;
    std::cout << Normalize(tprod2) << std::endl;
  }
