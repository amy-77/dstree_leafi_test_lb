cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET exec)
add_library(${TARGET} SHARED index.cc allocator.cc node.cc buffer.cc eapca.cc split.cc answer.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_14)
target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

target_link_libraries(${TARGET}
        PRIVATE filter util common ${Boost_LIBRARIES} ${TORCH_LIBRARIES} pthread
        )
