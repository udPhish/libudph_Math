#include <iostream>

#include <libudph/Math/World.h>

template<class T>
concept something = true;
auto calc(something auto f){
  return f;
}
int main(int argc, char** argv)
{
  auto m = UD::Tensor::PureTensor<int, 2>(2, 2);
  std::cout <<  calc(m) << std::endl;
  // auto m = UD::Tensor::Matrix<2, 2, int>{UD::Tensor::Vector<2, int>{0, 1},
  //                                        UD::Tensor::Vector<2, int>{2, 3}};

  // std::cout << m << std::endl;
  // auto s = UD::World::Shape<float, 2, 4, 4>{
  //    UD::Tensor::PureMatrix<float, 2, 4>(UD::T{1, 2}, {3, 4}),
  //    UD::Tensor::PureMatrix<UD::Math::Type::ULong, 2, 4>{1,
  //                                                        2,
  //                                                        2,
  //                                                        3,
  //                                                        3,
  //                                                        4,
  //                                                        4,
  //                                                        1}};
  // std::cout << s._points << std::endl;
  // std::cout << s._faces << std::endl;
}