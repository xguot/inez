# SPDX-License-Identifier: GPL-3.0-only
cmake_minimum_required(VERSION 3.20)
project(inezita VERSION 0.1.3 LANGUAGES CXX)

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
    "Inezita requires a C++20 compiler and standard library with std::span and "
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

add_library(inezita_options INTERFACE)
target_compile_features(inezita_options INTERFACE cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(inezita_options INTERFACE -Wall -Wextra -Wpedantic)
  if(INEZ_SANITIZERS)
    target_compile_options(inezita_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(inezita_options INTERFACE -fsanitize=address,undefined)
  endif()
elseif(INEZ_SANITIZERS)
  message(FATAL_ERROR "INEZ_SANITIZERS requires Clang or GCC")
endif()
add_library(inezita_chess
  src/chess.cpp
  src/encoding.cpp
  src/search.cpp
  src/uci.cpp
)
target_include_directories(inezita_chess PUBLIC include)
target_link_libraries(inezita_chess PUBLIC inezita_options Threads::Threads)

add_executable(inezita src/main.cpp)
target_compile_definitions(inezita PRIVATE INEZ_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(inezita PRIVATE inezita_chess)

add_executable(inezita-rules tools/rules.cpp)
target_link_libraries(inezita-rules PRIVATE inezita_chess)

add_executable(inezita-encode tools/encode.cpp)
target_link_libraries(inezita-encode PRIVATE inezita_chess)

if(BUILD_TESTING)
  add_executable(inezita-chess-tests tests/chess_tests.cpp)
  target_link_libraries(inezita-chess-tests PRIVATE inezita_chess)
  add_test(NAME chess COMMAND inezita-chess-tests)
  set_tests_properties(chess PROPERTIES TIMEOUT 120)

  add_executable(inezita-search-tests tests/search_tests.cpp)
  target_link_libraries(inezita-search-tests PRIVATE inezita_chess)
  add_test(NAME search COMMAND inezita-search-tests)
  set_tests_properties(search PROPERTIES TIMEOUT 120)

  add_executable(inezita-encoding-tests tests/encoding_tests.cpp)
  target_link_libraries(inezita-encoding-tests PRIVATE inezita_chess)
  add_test(NAME encoding COMMAND inezita-encoding-tests)
  set_tests_properties(encoding PROPERTIES TIMEOUT 120)

  if(INEZ_DIFFERENTIAL_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME differential
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/differential.py"
              "$<TARGET_FILE:inezita-rules>")
    set_tests_properties(differential PROPERTIES TIMEOUT 180)

    add_test(NAME baseline-uci
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/baseline_uci_tests.py"
              "$<TARGET_FILE:inezita>")
    set_tests_properties(baseline-uci PROPERTIES TIMEOUT 180)

    add_test(NAME contracts
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/contracts_tests.py"
              "$<TARGET_FILE:inezita-encode>")
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
      COMMAND "${CMAKE_COMMAND}" -E env "INEZ_BINARY=$<TARGET_FILE:inezita>"
              "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/neural_uci_tests.py")
    add_test(NAME neural-diagnostics
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/diagnostics_tests.py")
    add_test(NAME benchmark-integrity
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_tests.py")
    add_test(NAME trainer-replay
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/trainer_replay_tests.py")
    add_test(NAME promotion
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/promotion_tests.py")
    set_tests_properties(model training neural-uci neural-diagnostics benchmark-integrity trainer-replay promotion PROPERTIES TIMEOUT 180)
  endif()
endif()
