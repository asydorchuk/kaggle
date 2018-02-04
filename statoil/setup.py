'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='statoil',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Iceberg detection challenge',
      author='Slevin',
      author_email='iceberg@ocean.com',
      license='MIT',
      install_requires=[
          'keras==2.1.3',
          'h5py',
          'pandas==0.21.1',
          'numpy==1.13.3',
          'scikit-learn==0.18.2',
      ],
      zip_safe=False)
