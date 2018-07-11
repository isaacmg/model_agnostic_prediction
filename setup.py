from distutils.core import setup

setup(
    name='model_agnostic',
    version='0.1dev',
    packages=['model_agnostic', 'model_agnostic.models', 'model_agnostic.models.chexnet_files'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)