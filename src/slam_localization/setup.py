from setuptools import setup
import os
from glob import glob

package_name = 'slam_localization'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='KarmishthAnnauth',
    maintainer_email='karmishthannauth27@gmail.com',
    description='Localisation on a saved slam_toolbox map, republished as the particle filter pose.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pose_relay = slam_localization.pose_relay:main',
            'fit_start_pose = slam_localization.fit_start_pose:main',
        ],
    },
)
