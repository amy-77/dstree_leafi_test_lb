cmake_minimum_required(VERSION 3.16)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

#set(TEST_TENSOR test_tensor)
#
#add_executable(${TEST_TENSOR} test_tensor.cc logger.cc)
#set_property(TARGET ${TEST_TENSOR} PROPERTY CXX_STANDARD 14)
#
#target_compile_features(${TEST_TENSOR} PRIVATE cxx_std_14)
#target_link_libraries(${TEST_TENSOR} PRIVATE exec common util spdlog::spdlog ${TORCH_LIBRARIES})
#
#install(TARGETS ${TEST_TENSOR} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

#set(TEST_SPLINE test_gsl_spline)
#
#add_executable(${TEST_SPLINE} test_gsl_spline.cc)
#set_property(TARGET ${TEST_SPLINE} PROPERTY CXX_STANDARD 14)
#
#target_compile_features(${TEST_SPLINE} PRIVATE cxx_std_14)
#
#target_link_libraries(${TEST_SPLINE} PRIVATE GSL::gsl GSL::gslcblas)
#
#install(TARGETS ${TEST_SPLINE} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

####################
# speed test for cpu/gpu
####################
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TEST_CPU test_cpu)
add_executable(${TEST_CPU} test_model_cpu.cc)

target_link_libraries(${TEST_CPU} ${TORCH_LIBRARIES})
set_property(TARGET ${TEST_CPU} PROPERTY CXX_STANDARD 14)
#target_compile_features(${TEST_CPU} PRIVATE cxx_std_14)
#add_compile_options(-O0 -g -D_GNU_SOURCE -D_GLIBCXX_DEBUG)
#add_compile_options(-O0)

install(TARGETS ${TEST_CPU} DESTINATION ${CMAKE_SOURCE_DIR}/bin)

####################
# test a toy implementation for vmap
####################
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(TEST_VMAP test_vmap)
add_executable(${TEST_VMAP} test_vmap.cc)

target_link_libraries(${TEST_VMAP} ${TORCH_LIBRARIES})
set_property(TARGET ${TEST_VMAP} PROPERTY CXX_STANDARD 14)
#target_compile_features(${TEST_CPU} PRIVATE cxx_std_14)
#add_compile_options(-O0 -g -D_GNU_SOURCE -D_GLIBCXX_DEBUG)
#add_compile_options(-O0)

install(TARGETS ${TEST_VMAP} DESTINATION ${CMAKE_SOURCE_DIR}/bin)
