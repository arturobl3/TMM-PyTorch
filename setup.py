from setuptools import setup, find_packages

setup(
    name="TMM-PyTorch",
    version="0.1",
    packages=find_packages(),  # auto-discovers Python modules
    install_requires=[
        "python>=3.9",  # Python version requirement
        "torch>=2.0",  # PyTorch version requirement
    ],
)