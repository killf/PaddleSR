from setuptools import setup, find_packages

setup(
    name="PaddleSR",
    version="1.0",
    keywords=("SR", "Image", "Super-Resolution"),
    description="A paddle based toolkit  of image super-resolution for training or inference.",
    long_description="A paddle based toolkit of image super-resolution for training or inference.",
    license="Apache License 2.0",
    url="https://github.com/killf/PaddleSR",
    author="killf",
    author_email="killf@foxmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=['numpy', 'paddlepaddle-gpu', 'opencv-python', 'scikit-image'],
    scripts=[],
    entry_points={}
)