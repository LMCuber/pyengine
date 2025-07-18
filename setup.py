from setuptools import setup


setup(name="pyengine",
    version="0.1",
    description="Useful Python stuff",
    url="https://github.com/LMCuber/pyengine",
    author="Leo Bozkir",
    author_email="leo.bozkir@outlook.com",
    license="MIT",
    packages=[
        "pygame-ce",     # base module for games handling graphics, input, audio, etc.
        "pymunk",        # handling advanced 2D physics
        "pillow",        # some handy image manipulation functions 
        "opencv-python"  # for the skew function rotating a 2D texture in 3D space
    ]
    )
