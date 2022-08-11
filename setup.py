""" Sets up the Eigenvector Continuation library """
from setuptools import setup

setup(
    name='EigenvectorContinuation',
    url='https://github.com/kemperlab/EigenvectorContinuation',
    author='Jack H. Howard, Akhil Francis, Alexander F. Kemper',
    author_email='jhhoward@ncsu.edu',
    packages=['src.classes','src.util', 'src'],
    project_urls={
        "Documentation": "TODO https://github.com/kemperlab/EigenvectorContinuation", # TODO docs
        "Arxiv or DOI": "" # TODO Arxiv ro DOI
    },
    install_requires=['numpy','scipy'],
    version='1.0.1',
    license='BSD-2-Clause-Patent',
    description='Abstract class implementation of Eigenvector Continuation subspace diagonalization',
)
