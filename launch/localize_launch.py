# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # config and args
    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml'
    )
    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Localization configs')
    ld = LaunchDescription([localize_la])

    # nodes
    map_server_node = Node(
        package='map_server',
        executable='map_server',
        name='map_server_node',
        parameters=[LaunchConfiguration('localize_config')]
    )
    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter_node',
        parameters=[LaunchConfiguration('localize_config')]
    )

    # finalize
    ld.add_action(map_server_node)
    ld.add_action(pf_node)

    return ld