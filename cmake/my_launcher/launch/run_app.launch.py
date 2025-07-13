from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='your_python_package',
            executable='your_python_script',
            name='bool_array_publisher'
        ),
        Node(
            package='your_cpp_package',
            executable='bool_array_subscriber',
            name='bool_array_processor'
        )
    ])