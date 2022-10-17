cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TEST_SHARED_FROM_THIS test_shared_from_this)

add_executable(${TEST_SHARED_FROM_THIS} test_shared_from_this.cc)
set_property(TARGET ${TEST_SHARED_FROM_THIS} PROPERTY CXX_STANDARD 14)

target_compile_features(${TEST_SHARED_FROM_THIS} PRIVATE cxx_std_14)
target_link_libraries(${TEST_SHARED_FROM_THIS} PRIVATE exec util)

install(TARGETS ${TARGET_SHARED_FROM_THIS} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
