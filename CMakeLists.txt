cmake_minimum_required(VERSION 3.20)
set(TARGET "6502emu")
set(TARGET_COMMON "6502emu-lib")
set(TARGET_TEST "6502emu-test")
project(${TARGET} VERSION 0.0.1 LANGUAGES C CXX)

include(CTest)
enable_testing()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -pedantic)
add_compile_options(-g3)
#add_compile_options(-pg)
add_link_options(-g3)
#add_link_options(-pg)

set(SOURCES
    src/instruction.cpp
    src/cpu.cpp
    src/emulator.cpp
    src/control_flow.cpp
    src/codegen.cpp
    src/jit.cpp
)

set(TEST_SOURCES
    test/test-main.cpp
    test/test-jit.cpp
)

add_executable(${TARGET} src/main.cpp)
add_executable(${TARGET_TEST} ${TEST_SOURCES})
add_library(${TARGET_COMMON} STATIC ${SOURCES})

target_link_libraries(${TARGET} PRIVATE ${TARGET_COMMON})
target_link_libraries(${TARGET_TEST} PRIVATE ${TARGET_COMMON})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

find_package(fmt REQUIRED)
target_link_libraries(${TARGET_COMMON} PUBLIC fmt::fmt)

find_package(spdlog REQUIRED)
target_link_libraries(${TARGET_COMMON} PUBLIC spdlog::spdlog)

find_package(Boost COMPONENTS program_options REQUIRED)
target_link_libraries(${TARGET} PRIVATE Boost::program_options)

find_package(GTest REQUIRED)
target_link_libraries(${TARGET_TEST} PRIVATE GTest::GTest)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Find the libraries that correspond to the LLVM components that we wish to use
llvm_map_components_to_libnames(llvm_libs -16) # This is a hack!!

message(STATUS "LLVM libs: ${llvm_libs}")

# Link against LLVM libraries
target_link_libraries(${TARGET_COMMON} PUBLIC ${llvm_libs})

set_target_properties(${TARGET} ${TARGET_TEST} ${TARGET_COMMON} PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CXX_STANDARD 23
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${TARGET} PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
    target_compile_definitions(${TARGET_TEST} PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
    target_compile_definitions(${TARGET_COMMON} PRIVATE SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
endif()
