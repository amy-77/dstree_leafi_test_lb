cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET util)
add_library(${TARGET} SHARED config.cc logger.cc str.h global.h)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_14)
set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")

target_link_libraries(${TARGET}
        PUBLIC ${Boost_LIBRARIES}  ${TORCH_LIBRARIES}
        )
