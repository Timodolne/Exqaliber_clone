import unittest

from src.circuit_models.noise_model.base import NOISE_MODEL, BaseNoiseModel

class Depolarising(BaseNoiseModel):
    '''Depolarising noise
    
    Λ(ρ) = (1-p)ρ + (p/d)I

    Attributes
    ----------
    __p : float
        Probability the state depolarises
    __type : NOISE_MODEL.DEPOLARISING
        Depolarising noise model type

    Methods
    -------
    get_p()
        Get the probability that the state depolarises
    '''

    def __init__(self, p: float):
        """
        Parameters
        ----------
        p : float
            Probability that the state depolarises
        
        Raises
        ------
        ValueError:
            p must be within [0,1]
        """

        super().__init__(NOISE_MODEL.DEPOLARISING)
        if p < 0 or p > 1:
            raise ValueError(f"p = {p} is a probability and must be in [0,1]")
        self.__p = p
    
    def get_p(self) -> float:
        """Get probability state depolarises
        
        Returns
        -------
        float : 
            Probability that the state depolarises
        """

        return self.__p

class DepolarisingNoiseTestCase(unittest.TestCase):
    def test_p_bounds(self):
        invalid_p = (
            {'case': 'Negative p', 'p': -1},
            {'case': 'Large p', 'p': 2},
        )
        for case in invalid_p:
            with self.subTest(case['case']):
                self.assertRaises(ValueError,Depolarising,case['p'])
        
        with self.subTest("p within [0,1]"):
            self.assertEqual(0.5, Depolarising(0.5).get_p(), "Wrong probability loaded")
