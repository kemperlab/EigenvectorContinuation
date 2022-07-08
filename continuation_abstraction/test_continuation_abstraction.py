#%%
""" Tests continuation_abstraction interface methods.
    OUTDATED

    Supported:      NumpyArraySpace implementation
    Unsupported:    Other concrete implementations
"""
import unittest
import numpy as np
import continuation_abstraction as ca



class TestNumpyArraySpace(unittest.TestCase):
    """ tests the NumpyArraySpace implementation """


    # def setUp(self):
    #     """ initializes  """
    #     # Test Code: expects 4.j
    #     # vector1 = np.array([-1.j,3])
    #     # hamiltonian = np.array([[1,1.j],[-1.j,-1]])
    #     # vector2 = np.array([-3,-4.j])
    #     # evecs = np.array([[1,2],[7,6]])
    #     # hamiltonian = np.array([[1,0],[0,1]])
    #     # arrSpace = ca.NumpyArraySpace(evecs, hamiltonian)
    #     # print(arrSpace.expectation_value(vector1, hamiltonian, vector2))
    #     evecs = np.array([[1,0],[0,1]])
    #     hamiltonian = np.array([[1,1.j],[-1.j,-1]])
    #     arr_space = ca.NumpyArraySpace(evecs, hamiltonian)

    # def test_check_type_generic(self):
    #     """ tests the named method """

    #     evecs = np.array([0,0])
    #     hamiltonian = np.array([0,2],[2,0])
    #     arr_space = ca.NumpyArraySpace(evecs, hamiltonian)
    #     # try:
    #     # print(type(hamiltonian))
    #     arr_space.check_type_generic(hamiltonian)
    #     # except ValueError:
    #     #     raise ValueError from ValueError

    def test_expectation_value1(self):
        """ tests NumpyArraySpace """

        vector1 = np.array([-1.j,3])
        hamiltonian = np.array([[1,1.j],[-1.j,-1]])
        vector2 = np.array([-3,-4.j])
        evecs = np.array([[1,2],[7,6]]) # unused in calculation
        arr_space = ca.NumpyArraySpace(evecs, hamiltonian)


        self.assertEqual(22j, arr_space.expectation_value(vector1, hamiltonian, vector2))
        # self.assertEqual(22j, arr_space.inner_product(
        #             arr_space.inner_product(vector1, hamiltonian), vector2))

    def test_inner_product1(self):
        """ tests NumpyArraySpace """

        vector1 = np.array([1.j,3])
        vector2 = np.array([1,-4.j])
        # vector3 = np.array([[9],[0]])
        evecs = np.array([[1,2],[7,6]])             # unused in calculation
        hamiltonian = np.array([[1,-3.j],[-1.j,1]]) # unused in calculation
        arr_space = ca.NumpyArraySpace(evecs, hamiltonian)


        self.assertEqual(-13j, arr_space.inner_product(vector1, vector2))
        # try:
        #     arr_space.inner_product(vector2, vector3)
        #     self.assertEqual("Unexpected Exception", "")
        # except TypeError:
        #     pass

    def test_inner_product2(self):
        """ tests NumpyArraySpace """

        vector1 = np.array([2.j,3,1])
        vector2 = np.array([1,-1,1.j])
        # vector3 = np.array([[9],[0]])
        evecs = np.array([[1,2],[7,6]])             # unused in calculation
        hamiltonian = np.array([[1,-3.j],[-1.j,1]]) # unused in calculation
        arr_space = ca.NumpyArraySpace(evecs, hamiltonian)


        self.assertEqual(-3-1j, arr_space.inner_product(vector1, vector2))


    # # def test_check_basis_vecs_type(self, basis_vecs):
    # """ tests the named method """

    # # def test_check_ham_type(self, ham):
    # """ tests the named method """

    # # def test_inner_product(self, vec1, vec2):
    # """ tests the named method """

    # # def test_expectation_value(self, vec1, ham, vec2):
    # """ tests the named method """

    # def test_test(self):
    #     aaa = True
    #     bbb = True
    #     self.assertEqual(aaa, bbb)

if __name__ == '__main__':
    unittest.main()


#%%
