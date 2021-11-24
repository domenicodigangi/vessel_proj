from setuptools import find_packages, setup

setup(
    name='vessel_proj',
    packages= ["src"],
    package_dir={'src':'src'},
    version='0.1.0',
    description='Application of machine learning methods to AIS vessel data',
    author='Domenico Di Gangi',
    license='MIT',
)
