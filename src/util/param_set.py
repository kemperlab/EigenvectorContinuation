"""
    Sets up simple tuples to help code readability in other packages
"""

from collections import namedtuple

ParamSet = namedtuple("ParamSet", "j_x j_z b_x b_z")
"""" useful tuple when dealing with param sets in this space """
