cmake_minimum_required(VERSION 4.3.1)

project(MyStaticLibrary LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

find_package(Boost REQUIRED)
find_package(spdlog REQUIRED)
find_package(indicators REQUIRED)
find_package(magic_enum CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 REQUIRED)
find_package(Eigen3 REQUIRED)

option(BUILD_TESTS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set(RESOURCES_DIR "${CMAKE_BINARY_DIR}/resources")
file(MAKE_DIRECTORY ${RESOURCES_DIR})

#------------------------------------------------------- Libraries -----------------------------#
add_library(io src/cpp/src/io.cpp)
target_include_directories(io PRIVATE src/cpp/include ${Boost_INCLUDE_DIRS})
target_link_libraries(io PUBLIC nlohmann_json::nlohmann_json PRIVATE spdlog::spdlog sampler)

add_library(functionClasses src/cpp/src/functionClasses.cpp)
target_include_directories(functionClasses PRIVATE src/cpp/include)

add_library(odeModel src/cpp/src/odeModel.cpp)
target_include_directories(odeModel PUBLIC ${Boost_INCLUDE_DIRS} PRIVATE src/cpp/include)
target_link_libraries(odeModel PRIVATE functionClasses spdlog::spdlog io)

add_library(distributions src/cpp/src/distributions.cpp)
target_include_directories(distributions PUBLIC ${Eigen3_INCLUDE_DIRS} PRIVATE src/cpp/include ${Boost_INCLUDE_DIRS})
target_link_libraries(distributions PUBLIC Eigen3::Eigen)

add_library(sampler src/cpp/src/sampler.cpp src/cpp/src/proposalMechanisms.cpp)
target_include_directories(sampler PUBLIC ${indicators_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} PRIVATE src/cpp/include ${Boost_INCLUDE_DIRS})
target_link_libraries(sampler 
  PUBLIC 
  spdlog::spdlog 
  indicators::indicators 
  PRIVATE 
  odeModel 
  distributions 
  magic_enum::magic_enum 
  nlohmann_json::nlohmann_json)


#------------------------------------------------------- Experimens / Examples -----------------------------#

add_executable(thesis_example src/cpp/examples/thesis_example.cpp)
target_include_directories(thesis_example PRIVATE src/cpp/include ${Boost_INCLUDE_DIRS})
target_link_libraries(thesis_example PRIVATE functionClasses odeModel io sampler spdlog::spdlog magic_enum::magic_enum)

