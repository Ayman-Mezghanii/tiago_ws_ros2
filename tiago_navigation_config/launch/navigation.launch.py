import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare the launch arguments
    declare_use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (Gazebo) clock if true'
    )

    declare_map_yaml_file_arg = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(get_package_share_directory('tiago_navigation_config'), 'maps', 'my_map.yaml'),
        description='Full path to map file to load'
    )

    declare_param_file_arg = DeclareLaunchArgument(
        'param_file',
        default_value=os.path.join(get_package_share_directory('tiago_navigation_config'), 'config', 'nav2_params.yaml'),
        description='Full path to the ROS2 parameters file to use for all launched nodes'
    )

    # Get the launch directory
    bringup_dir = get_package_share_directory('nav2_bringup')

    # Include the bringup launch file from nav2_bringup
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(bringup_dir, 'launch', 'bringup_launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'map': LaunchConfiguration('map'),
            'params_file': LaunchConfiguration('param_file')
        }.items()
    )
# Add map_server node explicitly
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[LaunchConfiguration('param_file')],
        arguments=['--ros-args', '--param', 'yaml_filename:=LaunchConfiguration("map")']
    )
# Add static_transform_publisher node
    static_transform_publisher_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml_file = LaunchConfiguration('map')
    param_file = LaunchConfiguration('param_file')

    lifecycle_nodes = ['map_server', 'amcl', 'planner_server', 'controller_server', 'bt_navigator']
    # Create the LaunchDescription
    return LaunchDescription([
        declare_use_sim_time_arg,
        declare_map_yaml_file_arg,
        declare_param_file_arg,
        bringup_launch,
        map_server_node,
        static_transform_publisher_node,

Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[param_file]
        ),
        
        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[param_file]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[param_file]
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[param_file]
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[param_file]
        ),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': True},
                        {'node_names': lifecycle_nodes}]
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(
                get_package_share_directory('tiago_navigation_config'),
                'config',
                'rviz_config.rviz'
            )]
        ),
    ])


