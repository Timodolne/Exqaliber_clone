"""Implementation of a depolarising noise model."""
from exqaliber.circuit_sampling.noise_model.base import (
    NOISE_MODEL,
    BaseNoiseModel,
)


class Depolarising(BaseNoiseModel):
    """Depolarising noise.

    Λ(ρ) = (1-p)ρ + (p/d)I

    Attributes
    ----------
    p : float
        Probability the state depolarises
    type : NOISE_MODEL.DEPOLARISING
        Depolarising noise model type

    """

    def __init__(self, p: float):
        """Initialise Depolarising.

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
        self.p = p

    def get_p(self) -> float:
        """Get probability state depolarises.

        Returns
        -------
        float :
            Probability that the state depolarises

        """
        return self.p
