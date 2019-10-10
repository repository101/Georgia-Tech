
from setuptools import setup

setup(
    name='omscs',
    version='0.3.0',
    entry_points={
        'console_scripts': [
            'omscs = submit:main'
        ]
    },
    description='CLI for code submission to Udacity autograder for feedback',
    install_requires=[
        "nelson >= 0.4.2",
    ],
)