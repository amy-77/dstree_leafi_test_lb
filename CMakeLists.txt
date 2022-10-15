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

include_directories(
        ${Boost_INCLUDE_DIRS}
        src/exec
        src/util
)

add_subdirectory(src/exec)
add_subdirectory(src/util)

add_compile_options(-g)
target_compile_features(${TARGET} PRIVATE cxx_std_14)
set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")

target_link_libraries(${TARGET}
        PRIVATE exec util ${Boost_LIBRARIES}
        )

install(TARGETS ${TARGET} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
