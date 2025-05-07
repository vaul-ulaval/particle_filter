cmake_minimum_required(VERSION 3.16)
project(particle_filter)

# Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 14)

add_definitions(-DUSE_CUDA=1)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(CUDA REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Include project headers
include_directories(
  include
  ${CMAKE_INSTALL_PREFIX}/include/range_libc
  ${CMAKE_INSTALL_PREFIX}/include/range_libc/vendor
)

# Link directories
link_directories(
  ${CMAKE_INSTALL_PREFIX}/lib
)



add_executable(particle_filter src/particle_filter.cpp)

# # Set compile options
# target_compile_features(particle_filter PRIVATE cxx_std_17)

target_link_libraries(particle_filter
  range_libc
)

ament_target_dependencies(particle_filter
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  nav_msgs
  geometry_msgs
  tf2_geometry_msgs
)

target_include_directories(particle_filter PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(TARGETS particle_filter
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)


ament_export_include_directories(include)

ament_package()