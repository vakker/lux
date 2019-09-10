from setuptools import find_packages, setup

setup(
    name='pytorch-ray',
    version='0.1',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'torch==1.2.0',
        'ray==0.8.0.dev4',
        'pandas>=0.20.3',
    ],
)
