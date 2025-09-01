#
# Created by Qitong Wang on 2022/10/1.
# Copyright (c) 2022 Université Paris Cité. All rights reserved.
#

# 在文件开始部分添加全局PIC设置
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_minimum_required(VERSION 3.16)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(dstree CXX)

# 添加可执行文件
set(TARGET dstree)
add_executable(${TARGET} main.cc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CONFIG_LIB_CPLUSPLUS ON)
#set(CONFIG_NEWLIB_LIBC ON)

find_package(Boost COMPONENTS math_c99 program_options filesystem system REQUIRED)

# 设置 GSL 根目录（使用当前 conda 环境路径）
set(GSL_ROOT_DIR "$ENV{CONDA_PREFIX}")

# 显式指定头文件和库文件路径
set(GSL_INCLUDE_DIRS "${GSL_ROOT_DIR}/include")
set(GSL_LIBRARIES "${GSL_ROOT_DIR}/lib/libgsl.so" "${GSL_ROOT_DIR}/lib/libgslcblas.so")

# 配置FindGSL.cmake，使其跳过默认查找逻辑
set(GSL_FOUND TRUE)
set(GSL_CBLAS_FOUND TRUE)
set(GSL_VERSION "2.7")

find_package(GSL REQUIRED)

list(APPEND CMAKE_MODULE_PATH "$ENV{LIBTORCH_ROOT}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_ROOT}/share/cmake/Torch")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${GSL_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        src/common
        src/util
        src/model
        src/filter
        src/navigator
        src/exec
)

# 添加Eigen头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

add_subdirectory(lib/spdlog-1.11.0)
#add_subdirectory(lib/fmt-9.1.0)

add_subdirectory(src/common)
add_subdirectory(src/util)
add_subdirectory(src/model)
add_subdirectory(src/filter)
add_subdirectory(src/navigator)
add_subdirectory(src/exec)

#add_compile_options(-O3 -Werror)
#add_compile_options(-O0 -g -D_GNU_SOURCE -D_GLIBCXX_DEBUG -lasan -fsanitize=address -fsanitize=undefined)
add_compile_options(-O0 -g -D_GNU_SOURCE -D_GLIBCXX_DEBUG)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 链接所有依赖库
target_link_libraries(${TARGET}
        PRIVATE exec navigator filter util common spdlog::spdlog
        ${GSL_LIBRARIES}   # 添加 GSL 库
        ${Boost_LIBRARIES} # 确保 Boost 库已链接
        ${TORCH_LIBRARIES} # 确保 Torch 库已链接
        )

install(TARGETS ${TARGET} DESTINATION ${PROJECT_SOURCE_DIR}/bin)

#add_subdirectory(test)
