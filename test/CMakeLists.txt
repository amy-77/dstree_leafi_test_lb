cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(TEST_SHARED_FROM_THIS test_shared_from_this)
#
#add_executable(${TEST_SHARED_FROM_THIS} test_shared_from_this.cc)
#set_property(TARGET ${TEST_SHARED_FROM_THIS} PROPERTY CXX_STANDARD 14)
#
#target_compile_features(${TEST_SHARED_FROM_THIS} PRIVATE cxx_std_14)
#target_link_libraries(${TEST_SHARED_FROM_THIS} PRIVATE exec util)
#
#install(TARGETS ${TARGET_SHARED_FROM_THIS} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

#set(TEST_SIMD test_simd)
#
#add_executable(${TEST_SIMD} test_simd.cc)
#set_property(TARGET ${TEST_SIMD} PROPERTY CXX_STANDARD 14)
#
#target_compile_features(${TEST_SIMD} PRIVATE cxx_std_14)
#target_link_libraries(${TEST_SIMD} PRIVATE common util ${Boost_LIBRARIES})
#
#install(TARGETS ${TEST_SIMD} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

set(TEST_TENSOR test_tensor)

add_executable(${TEST_TENSOR} test_tensor.cc logger.cc)
set_property(TARGET ${TEST_TENSOR} PROPERTY CXX_STANDARD 14)

target_compile_features(${TEST_TENSOR} PRIVATE cxx_std_14)
target_link_libraries(${TEST_TENSOR} PRIVATE exec common util spdlog::spdlog ${TORCH_LIBRARIES})

install(TARGETS ${TEST_TENSOR} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
