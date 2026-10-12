import os
from glob import glob

from setuptools import find_packages, setup

package_name = "minddrive_f1tenth"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="MindDrive VLA on a real F1TENTH car (Orin inference, Nano actuation).",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "minddrive_realworld_node = minddrive_f1tenth.minddrive_realworld_node:main",
        ],
    },
)
