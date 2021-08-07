# CMAKE VERSION
cmake_minimum_required(VERSION 3.18.0)

include(cmake/udph.cmake)

project(libudph_Math VERSION "${GIT_VERSION}" LANGUAGES CXX)

include(udphInit)

PROJECT_SET_NAMESPACE(libudph_Math libudph)

TARGET_CREATE_LIB(Math INTERFACE)
TARGET_APPEND_HDR(
	Math
	"libudph/Math/Tensor.h"
	"libudph/Math/Math.h"
)

TARGET_LINK(Math)

TARGET_PACKAGE(Math)

include(udphFinalize)