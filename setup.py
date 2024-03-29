""" Sets up the Eigenvector Continuation library fro Python 3.8 """
from setuptools import setup

setup(
    name='EigenvectorContinuation',
    url='https://github.com/kemperlab/EigenvectorContinuation',
    author='Jack H. Howard, Akhil Francis, Alexander F. Kemper',
    author_email='jhhoward@ncsu.edu',
    packages=['eigenvectorcontinuation.continuer','eigenvectorcontinuation.hilbertspaces', 'eigenvectorcontinuation.util','eigenvectorcontinuation'],
    project_urls={
        "Documentation": "TODO https://github.com/kemperlab/EigenvectorContinuation", # TODO docs
        "Arxiv or DOI": "" # TODO Arxiv ro DOI
    },
    install_requires=['numpy','scipy','matplotlib'],
    version='1.0',
    license='BSD-2-Clause-Patent',
    description='Abstract class implementation of Eigenvector Continuation subspace diagonalization',
)
