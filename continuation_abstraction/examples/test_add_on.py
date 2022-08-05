#%%
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from src.continuation_abstraction import HilbertSpaceAbstract as hsa


# TODO: figure out this import^, make the mimic, and run it

print(hsa.basis_vecs)


#%%