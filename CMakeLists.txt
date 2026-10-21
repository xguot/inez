# SPDX-License-Identifier: GPL-3.0-only
cmake_minimum_required(VERSION 3.20)
project(inez VERSION 0.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Older compilers may accept a C++20 flag without implementing the features
# used by the engine. Check the compiler and its standard library together.
include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[
  #include <span>
  struct Value {
    int value;
    bool operator==(const Value &) const = default;
  };
  static_assert(Value{1} == Value{1});
  int main() {
    int values[] = {1, 2};
    return std::span<const int>(values).size() == 2 ? 0 : 1;
  }
]] INEZ_HAS_REQUIRED_CXX20)
if(NOT INEZ_HAS_REQUIRED_CXX20)
  message(FATAL_ERROR
    "Inez requires a C++20 compiler and standard library with std::span and "
    "defaulted comparisons. Selected compiler: ${CMAKE_CXX_COMPILER} "
    "(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}). "
    "Load a modern compiler module (on Rivanna: module load gcc; "
    "use module spider gcc to list available versions), "
    "then configure a fresh build directory with CXX set to that compiler.")
endif()

include(CTest)
option(INEZ_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(INEZ_DIFFERENTIAL_TESTS "Run tests against the Python chess oracle" OFF)
option(INEZ_MODEL_TESTS "Run optional PyTorch architecture/artifact tests" OFF)

find_package(Threads REQUIRED)

add_library(inez_options INTERFACE)
target_compile_features(inez_options INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(inez_options INTERFACE -Wall -Wextra -Wpedantic)
  if(INEZ_SANITIZERS)
    target_compile_options(inez_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(inez_options INTERFACE -fsanitize=address,undefined)
  endif()
elseif(INEZ_SANITIZERS)
  message(FATAL_ERROR "INEZ_SANITIZERS requires Clang or GCC")
endif()
add_library(inez_chess
  src/chess.cpp
  src/encoding.cpp
  src/search.cpp
  src/uci.cpp
)
target_include_directories(inez_chess PUBLIC include)
target_link_libraries(inez_chess PUBLIC inez_options Threads::Threads)

add_executable(inez src/main.cpp)
target_link_libraries(inez PRIVATE inez_chess)

add_executable(inez-rules tools/rules.cpp)
target_link_libraries(inez-rules PRIVATE inez_chess)

add_executable(inez-encode tools/encode.cpp)
target_link_libraries(inez-encode PRIVATE inez_chess)

if(BUILD_TESTING)
  add_executable(inez-chess-tests tests/chess_tests.cpp)
  target_link_libraries(inez-chess-tests PRIVATE inez_chess)
  add_test(NAME chess COMMAND inez-chess-tests)
  set_tests_properties(chess PROPERTIES TIMEOUT 120)

  add_executable(inez-search-tests tests/search_tests.cpp)
  target_link_libraries(inez-search-tests PRIVATE inez_chess)
  add_test(NAME search COMMAND inez-search-tests)
  set_tests_properties(search PROPERTIES TIMEOUT 120)

  add_executable(inez-encoding-tests tests/encoding_tests.cpp)
  target_link_libraries(inez-encoding-tests PRIVATE inez_chess)
  add_test(NAME encoding COMMAND inez-encoding-tests)
  set_tests_properties(encoding PROPERTIES TIMEOUT 120)

  if(INEZ_DIFFERENTIAL_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME differential
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/differential.py"
              "$<TARGET_FILE:inez-rules>")
    set_tests_properties(differential PROPERTIES TIMEOUT 180)

    add_test(NAME match
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/match.py"
              "$<TARGET_FILE:inez>" --pairs 2 --nodes 48 --max-plies 100)
    set_tests_properties(match PROPERTIES TIMEOUT 180)

    add_test(NAME contracts
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/contracts_tests.py"
              "$<TARGET_FILE:inez-encode>")
    add_test(NAME data
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/data_tests.py")
    set_tests_properties(contracts data PROPERTIES TIMEOUT 180)
  endif()

  if(INEZ_MODEL_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME model
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/model_tests.py")
    add_test(NAME training
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/training_tests.py")
    add_test(NAME neural-uci
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/neural_uci_tests.py")
    add_test(NAME neural-diagnostics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/diagnostics_tests.py")
    set_tests_properties(model training neural-uci neural-diagnostics PROPERTIES TIMEOUT 180)
  endif()
endif()
