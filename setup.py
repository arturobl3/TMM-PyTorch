from setuptools import setup, find_packages

setup(
    name="TMM-PyTorch",
    version="0.1",
    packages=find_packages(),
    python_requires='>=3.10',  
    install_requires=[
        "torch>=2.0",
        "numpy>=1.21",
    ],
)