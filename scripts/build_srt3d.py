"""Build the pinned native SRT3D library and this repository's interactive adapter."""

import argparse
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--build", type=Path, required=True)
p.add_argument("--jobs", type=int, default=4)
a = p.parse_args()
source, build = a.source.resolve(), a.build.resolve()
subprocess.run(
    [
        "cmake",
        "-S",
        str(source / "SRT3D"),
        "-B",
        str(build),
        "-DUSE_AZURE_KINECT=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ],
    check=True,
)
subprocess.run(["cmake", "--build", str(build), "--target", "srt3d", "-j", str(a.jobs)], check=True)
flags = subprocess.check_output(
    ["pkg-config", "--cflags", "--libs", "opencv4", "glfw3", "glew"], text=True
).split()
subprocess.run(
    [
        "g++",
        "-std=c++17",
        "-O3",
        "-fopenmp",
        "-I" + str(source / "SRT3D/include"),
        "-I/usr/include/eigen3",
        str(Path(__file__).resolve().parents[1] / "native/srt3d_tracker.cpp"),
        str(build / "src/libsrt3d.a"),
        "-o",
        str(build / "spacergb_srt3d"),
        *flags,
        "-lOpenGL",
    ],
    check=True,
)
print(build / "spacergb_srt3d")
