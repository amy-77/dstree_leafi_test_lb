cmake_minimum_required(VERSION 3.16)

set(TARGET exec)
add_library(${TARGET} SHARED index.cc node.cc filter.cc buffer.cc eapca.cc split.cc answer.cc stat.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_features(${TARGET} PRIVATE cxx_std_14)
#target_compile_options(${TARGET} PRIVATE -O3 -ffast-math)

target_link_libraries(${TARGET}
        PRIVATE util ${Boost_LIBRARIES}
        )
