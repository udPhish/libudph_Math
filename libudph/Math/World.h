#pragma once
#include <libudph/Class/Interface.h>
#include <libudph/Math/Tensor.h>

namespace UD::World
{
template<class _Type,
         Math::Type::ULong _Dimension,
         Math::Type::ULong _Points,
         Math::Type::ULong _Faces>
// requires(_Dimension == 2)
struct Shape
    : public Interface::Interface<Shape<_Type, _Dimension, _Points, _Faces>>
{
  Tensor::PureMatrix<_Type, _Dimension, _Points>            _points = {};
  Tensor::PureMatrix<Math::Type::ULong, _Dimension, _Faces> _faces  = {};

  Shape(Tensor::PureMatrix<_Type, _Dimension, _Points>            points,
        Tensor::PureMatrix<Math::Type::ULong, _Dimension, _Faces> faces)
      : _points{points}
      , _faces{faces}
  {
  }
};
}  // namespace UD::World