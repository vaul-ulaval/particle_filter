from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("particle_filter"),
        "config",
        "particle_filter.yaml",
    )
    return LaunchDescription([
        Node(
            package='particle_filter',
            executable='particle_filter',
            name='particle_filter',
            output='screen',
            parameters=[
                config
            ]
        )
    ])
