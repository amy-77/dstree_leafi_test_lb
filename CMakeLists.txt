#
# Created by Qitong Wang on 2022/10/1.
# Copyright (c) 2022 Université Paris Cité. All rights reserved.
#

cmake_minimum_required(VERSION 3.16)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(dstree CXX)

set(TARGET dstree)
add_executable(${TARGET} main.cc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#set(CONFIG_LIB_CPLUSPLUS ON)
#set(CONFIG_NEWLIB_LIBC ON)

find_package(Boost 1.80.0 COMPONENTS program_options log log_setup filesystem system REQUIRED)

list(APPEND CMAKE_MODULE_PATH "$ENV{LIBTORCH_ROOT}/share/cmake/Torch")
list(APPEND CMAKE_PREFIX_PATH "$ENV{LIBTORCH_ROOT}/share/cmake/Torch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
        src/common
        src/util
        src/filter
        src/exec
)

add_subdirectory(lib/spdlog-1.11.0)
#add_subdirectory(lib/fmt-9.1.0)

add_subdirectory(src/common)
add_subdirectory(src/util)
add_subdirectory(src/filter)
add_subdirectory(src/exec)

add_compile_options(-O0)
#add_compile_options(-g -Werror -O3)
#add_compile_options(-fsanitize=address -fsanitize=undefined -D_GLIBCXX_DEBUG)

target_compile_features(${TARGET} PRIVATE cxx_std_14)

target_link_libraries(${TARGET}
        PRIVATE exec filter util common spdlog::spdlog #fmt::fmt
        )

install(TARGETS ${TARGET} DESTINATION ${PROJECT_SOURCE_DIR}/bin)

#add_subdirectory(test) # not updated
