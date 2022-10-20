cmake_minimum_required(VERSION 3.16)

set(TARGET common)
add_library(${TARGET}
        SHARED common.h stat.h distance.h)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_features(${TARGET} PRIVATE cxx_std_14)

target_link_libraries(${TARGET}
        PRIVATE util
        )
