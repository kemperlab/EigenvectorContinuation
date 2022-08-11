""" Sets up the Eigenvector Continuation library """
from setuptools import setup

setup(
    name='ec_abstract',
    url='https://github.com/kemperlab/EigenvectorContinuation',
    author='Jack Howard, Akhil Francis',
    author_email='jhhoward@ncsu.edu, afranci2@ncsu.edu',
    packages=['src.classes','src.util', 'src'],
    project_urls={
        "Documentation": "TODO https://github.com/kemperlab/EigenvectorContinuation",
    },
    install_requires=['numpy','scipy'],
    version='0.1',
    license='BSD-2-Clause Plus Patent License',
    description='Abstract class implementation of Eigenvector Continuation subspace diagonalization',
)
