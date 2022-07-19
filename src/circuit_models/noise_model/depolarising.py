from circuit_models.noise_model.base import NOISE_MODEL, BaseNoiseModel

class Depolarising(BaseNoiseModel):
    '''Depolarising noise
    
    Λ(ρ) = (1-p)ρ + (p/d)I

    Attributes
    ----------
    __p : float
        Probability the state depolarises
    __type : NOISE_MODEL.DEPOLARISING
        Depolarising noise model type

    '''

    def __init__(self, p: float):
        """
        Parameters
        ----------
        p : float
            Probability that the state depolarises
        """

        super().__init__(NOISE_MODEL.DEPOLARISING)
        self.__p = p
    
    def get_p(self) -> float:
        """Get probability state depolarises
        
        Returns
        -------
        float : 
            Probability that the state depolarises
        """

        return self.__p
