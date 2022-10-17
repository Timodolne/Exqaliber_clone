"""Implementation of a depolarising noise model."""
import numpy as np

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

    def __init__(self, p: float = None, L: float = None):
        """Initialise Depolarising.

        Parameters
        ----------
        p : float, optional
            Probability that the state depolarises, value is inferred
            from `L` if not provided
        L : float, optional
            Rate of depolarisation, value is inferred from `p` if not
            provided

        Raises
        ------
        ValueError:
            If given values of `p`, `L` don't match or are outside the
            ranges 0 < `p` < 1, `L` > 0
        AttributeError:
            If neither `p` nor `L` is provided.
        """
        super().__init__(NOISE_MODEL.DEPOLARISING)

        if p and L:
            if L != -np.log(1 - p):
                raise ValueError(
                    "Provided values for p and L that don't match!"
                    "Should have 1 - p = e^{-L}"
                )
            self.p, self.L = p, L
        elif p:
            if p < 0 or p > 1:
                raise ValueError(
                    f"p = {p} is a probability and must be in [0,1]"
                )
            self.p = p
            self.L = -np.log(1 - p)
        elif L:
            if L <= 0:
                raise ValueError("L must be a positive number.")
            self.p = 1 - np.exp(-L)
        else:
            raise AttributeError("No value for p or L provided")

    def get_p(self) -> float:
        """Get probability state depolarises.

        Returns
        -------
        float :
            Probability that the state depolarises, `self.p`
        """
        return self.p

    def get_L(self) -> float:
        """Get exponential parameter of depolarising channel.

        Returns
        -------
        float
            Rate of depolarisation, `self.L`
        """
        return self.L

    def get_coherence_probability(self, n: int = 0):
        """Get the probability that the state has not decohered.

        Parameters
        ----------
        n : int, optional
            Number of times the channel is used, by default 0

        Returns
        -------
        float
            Probability that the state remains coherent after
            `n` uses of the channel
        """
        return np.power(1 - self.p, n)

    def get_decoherence_probability(self, n: int = 0):
        """Get the probability that the state has decohered.

        Parameters
        ----------
        n : int, optional
            Number of times the channel is used, by default 0

        Returns
        -------
        float
            Probability that the state remains coherent after
            `n` uses of the channel
        """
        return 1 - self.get_coherence_probability(n)
