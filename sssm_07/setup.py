from setuptools import setup
from setuptools import find_packages

setup(
    name='sssm',
    version='0.0.7',
    packages=find_packages(),
    include_package_data=True,
#    package_data={"saved_models": ["*.pt"]},
    install_requires=[
        'requests',
        'importlib-metadata',
        'torch',
        'pandas',
        'einops',
        'seaborn',
        'numpy',
        'scipy',
        'matplotlib',
    ],
)
