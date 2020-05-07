from setuptools import setup, find_packages

setup(
    name                 = 'noisypy',
    version              = '0.1',
    description          = '''''',
    long_description     = (''.join(open('README.md').readlines())),
    install_requires     = ['numpy', 'scipy', 'pandas', 'csaps'],
    license              = 'License :: MIT',
    packages             = find_packages(),
    package_data         = {'noisypy': ['config.cfg']},
    )