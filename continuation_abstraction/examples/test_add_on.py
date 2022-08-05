#%%
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import eigh
from scipy.linalg import null_space
from matplotlib import pyplot as plt
from continuation_abstraction import continuation_abstraction as ca



#%%