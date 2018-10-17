from setuptools import setup
from setuptools import find_packages


setup(
    name='model_agnostic',
    version='0.37dev',
    packages=['model_agnostic', 'model_agnostic.models', 'model_agnostic.example_models', 'model_agnostic.example_models.chexnet_files'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description='A package to easily serve machine learning models',

)
