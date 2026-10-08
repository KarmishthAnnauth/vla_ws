import os
from glob import glob

from setuptools import find_packages, setup

package_name = "simlingo_f1tenth"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="SimLingo VLA on a real F1TENTH car (Orin inference, Nano actuation).",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simlingo_realworld_node    = simlingo_f1tenth.simlingo_realworld_node:main",
            "trajectory_controller_node = simlingo_f1tenth.trajectory_controller_node:main",
            "camera_node                = simlingo_f1tenth.camera_node:main",
        ],
    },
)
