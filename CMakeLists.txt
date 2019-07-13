cmake_minimum_required (VERSION 3.0)

project(BNN)

add_subdirectory(eigen-git-mirror)
add_subdirectory(NN)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable("${PROJECT_NAME}" main.cpp)

target_link_libraries("${PROJECT_NAME}" PRIVATE Eigen3::Eigen NN)