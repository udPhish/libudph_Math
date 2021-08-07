# CMAKE VERSION
cmake_minimum_required(VERSION 3.18.0)

include(cmake/udph.cmake)

project(libudph VERSION "${GIT_VERSION}" LANGUAGES CXX)

include(udphInit)

TARGET_CREATE_LIB(SomeLib INTERFACE)
TARGET_APPEND_HDR(SomeLib "libudph/somelib/just.h")

target_create_exe(SomeTarget)
target_append_src(SomeTarget "libudph/SomeTarget.cpp")

TARGET_APPEND_DEP(SomeTarget SomeLib)

TARGET_LINK(SomeLib)
target_link(SomeTarget)

target_package(SomeLib)
target_package(SomeTarget)

include(udphFinalize)