# SPDX-License-Identifier: GPL-3.0-only
cmake_minimum_required(VERSION 3.20)
project(inez VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
option(INEZ_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(INEZ_DIFFERENTIAL_TESTS "Run tests against the Python chess oracle" OFF)

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
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(inez_chess
  src/chess.cpp
  src/search.cpp
  src/uci.cpp
)
target_include_directories(inez_chess PUBLIC include)
target_link_libraries(inez_chess PUBLIC inez_options Threads::Threads)

add_executable(inez src/main.cpp)
target_link_libraries(inez PRIVATE inez_chess)

add_executable(inez-rules tools/rules.cpp)
target_link_libraries(inez-rules PRIVATE inez_chess)

if(BUILD_TESTING)
  add_executable(inez-chess-tests tests/chess_tests.cpp)
  target_link_libraries(inez-chess-tests PRIVATE inez_chess)
  add_test(NAME chess COMMAND inez-chess-tests)
  set_tests_properties(chess PROPERTIES TIMEOUT 120)

  add_executable(inez-search-tests tests/search_tests.cpp)
  target_link_libraries(inez-search-tests PRIVATE inez_chess)
  add_test(NAME search COMMAND inez-search-tests)
  set_tests_properties(search PROPERTIES TIMEOUT 120)

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
  endif()
endif()
