# CMAKE VERSION
cmake_minimum_required(VERSION 3.18.0)

include(cmake/udph.cmake)

project_create(libudph_Math)

PROJECT_SET_NAMESPACE(libudph_Math libudph)

project_load_git_repository(libudph_Class "https://github.com/udPhish/libudph_Class.git" master)
project_load_git_repository(libudph_Container "https://github.com/udPhish/libudph_Container.git" master)

TARGET_CREATE_LIB(Math STATIC)
TARGET_APPEND_SRC(
	Math
	"libudph/Math/dummy.cpp"
)
TARGET_APPEND_HDR(
	Math
	"libudph/Math/Tensor.h"
	"libudph/Math/udMath.h"
)
TARGET_APPEND_DEP(Math libudph::Class)
TARGET_APPEND_DEP(Math libudph::Container)

TARGET_LINK(Math)

FINALIZE()