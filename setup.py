from setuptools import setup, find_packages

setup(
    name='platy_reg',
    version='1.1',
    description='Registration of platynereis fluorescence and EM volumes.',
    url='https://github.com/kreshuklab/platy_registration',
    packages=find_packages(include=['platy_reg']),
    python_requires='>=3.6',
    install_requires=[],
    author='Elena Buglakova',
    author_email='elena.v.buglakova@gmail.com',
    license='MIT'
)
