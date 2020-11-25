from setuptools import setup

setup(
    name='deepchall',
    version='0.1',
    author='Luigi Coniglio',
    description='Benchmarking tool for deep neural networks',
    packages=[
              'deepchall', 
              'deepchall.backends', 
              'deepchall.langs',
              'deepchall.nets',
    ],
    install_requires=[
        'click',
    ],
    entry_points='''
        [console_scripts]
        deepchall=deepchall.cli:cli
    ''',
)
