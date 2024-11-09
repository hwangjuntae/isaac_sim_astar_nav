# launch/astar_nav.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Launch 인자 선언: use_sim_time을 True로 고정
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true'
    )

    # RViz2 구성 파일 경로 설정
    rviz_config_dir = ('/home/teus/tb3_ws/src/astar/rviz/astar_config.rviz')

    # A* Navigation Node 실행
    astar_node = Node(
        package='astar',  # 실제 패키지 이름으로 변경
        executable='astar.py',  # 실제 실행 파일 이름으로 변경
        name='astar_navigation',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    # Static Transform Publisher 실행 (map -> odom)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_to_odom',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
    )

    # RViz2 실행
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_dir],
        output='screen',
    )

    return LaunchDescription([
        use_sim_time_arg,
        astar_node,
        static_tf_node,
        rviz_node,
    ])
