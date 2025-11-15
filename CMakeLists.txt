cmake_minimum_required(VERSION 3.5)
project(particle_filter)

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add CUDA support definition for range_libc compatibility
add_definitions(-DUSE_CUDA=1)

find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

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

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/particle_filter.cpp
)

target_link_libraries(${PROJECT_NAME}
  range_libc
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN particle_filter::ParticleFilter
  EXECUTABLE particle_filter_node
)


target_include_directories(particle_filter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

install(DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}/
)


ament_export_include_directories(include)

ament_auto_package()