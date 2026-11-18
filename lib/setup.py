# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup

setup(
    name="faster_rcnn",
    version="0.1",
    description="object detection in pytorch",
    packages=find_packages(exclude=("configs", "tests",)),
)
