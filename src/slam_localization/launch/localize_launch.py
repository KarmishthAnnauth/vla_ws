"""Localise on a saved slam_toolbox pose graph and publish the pose on /pf/pose/odom.

Run with bringup already up and the car in the start box, nobody next to it:
    ros2 launch slam_localization localize_launch.py [map:=track_20260930] [start_pose:=auto|x,y,yaw] [smoothing_time:=0.3] [udp_only:=true]
The map name refers to <name>.posegraph/.data in particle_filter/maps.

start_pose is the base_link pose in the map frame (metres, radians). The default 'auto' fits
it from the current /scan (fit_start_pose) before slam_toolbox starts; the start box is
searched, so the car must be there. It must be within about 0.5 m of the truth, because
slam_toolbox only corrects it once the car moves. 0,0,0 is the pose bringup had when the
map was recorded.

udp_only:=true (default) runs these nodes, and the start-pose fit, with Fast DDS over UDP only
(config/fastdds_udp_only.xml). Stale shared-memory files from crashed processes otherwise make
newly started nodes miss /scan, /odom or /tf at random. They still talk to bringup normally.

Do not run this together with slam_toolbox mapping or particle_filter: all of them own
the map frame.
"""
import os
import subprocess
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def fit_start_pose(map_name, env):
    """Run fit_start_pose and return ([x, y, yaw], log); raises RuntimeError with its output on failure."""
    try:
        res = subprocess.run(['ros2', 'run', 'slam_localization', 'fit_start_pose', '--map', map_name],
                             capture_output=True, text=True, timeout=90, env=env)
    except subprocess.TimeoutExpired:
        raise RuntimeError('fit_start_pose timed out: is bringup running and /scan publishing?')
    for line in res.stdout.splitlines():
        if line.startswith('start_pose:='):
            return [float(v) for v in line.split('=', 1)[1].split(',')], res.stderr.strip()
    raise RuntimeError('fit_start_pose failed:\n' + res.stdout + res.stderr)


def nodes(context):
    map_name = LaunchConfiguration('map').perform(context)
    map_path = os.path.join(get_package_share_directory('particle_filter'), 'maps', map_name)
    params = os.path.join(get_package_share_directory('slam_localization'), 'config', 'localization.yaml')
    start_arg = LaunchConfiguration('start_pose').perform(context)
    actions = []
    env = dict(os.environ)
    if LaunchConfiguration('udp_only').perform(context).lower() in ('true', '1', 'yes'):
        profile = os.path.join(get_package_share_directory('slam_localization'), 'config', 'fastdds_udp_only.xml')
        env['FASTRTPS_DEFAULT_PROFILES_FILE'] = profile
        actions.append(SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', profile))
    if start_arg == 'auto':
        start_pose, fit_log = fit_start_pose(map_name, env)
        actions.append(LogInfo(msg='fit_start_pose: %s -> start_pose %.3f,%.3f,%.4f' % (fit_log, *start_pose)))
    else:
        start_pose = [float(v) for v in start_arg.split(',')]
    return actions + [
        # slam_toolbox's base_frame is base_footprint (the stock config that works on this car)
        Node(package='tf2_ros', executable='static_transform_publisher',
             name='static_baselink_to_footprint',
             arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint']),
        Node(package='slam_toolbox', executable='localization_slam_toolbox_node',
             name='slam_toolbox', output='screen',
             parameters=[params, {'map_file_name': map_path, 'map_start_pose': start_pose, 'use_sim_time': False}]),
        Node(package='slam_localization', executable='pose_relay', name='pose_relay', output='screen',
             parameters=[{'smoothing_time': float(LaunchConfiguration('smoothing_time').perform(context))}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='track_20260930',
                              description='pose graph name in particle_filter/maps'),
        DeclareLaunchArgument('start_pose', default_value='auto',
                              description="initial base_link pose in the map: 'auto' (fit from /scan) or x,y,yaw"),
        DeclareLaunchArgument('udp_only', default_value='true',
                              description='use Fast DDS over UDP only for these nodes (avoids stale shared-memory files)'),
        DeclareLaunchArgument('smoothing_time', default_value='0.3',
                              description='pose_relay: seconds over which slam_toolbox corrections are blended in (0 = raw)'),
        OpaqueFunction(function=nodes),
    ])
