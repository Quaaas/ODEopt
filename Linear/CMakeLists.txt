project(Spacy-Examples-EigenAndFung-Example1)

cmake_minimum_required(VERSION 3.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include path, link path, library and compiler flags for Spacy
find_package(Spacy REQUIRED)
# include_directories(${Spacy_INCLUDE_DIR})

include_directories("/home/sebastian/Documents/Master/Spacy/")
add_definitions(${Spacy_CXX_FLAGS})


# Executable
add_executable(run main.cpp ../util.hh ../util.cpp ../ODEopt.hh ../ODEopt.cpp)

# Target libraries
#target_link_libraries(run ${Spacy_LIBRARIES})
target_link_libraries(run "/home/sebastian/Documents/Master/Spacy/build/Spacy/libspacy.a")
