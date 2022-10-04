"""Implementation of a noiseless noise model."""
from exqaliber.circuit_sampling.noise_model.base import (
    NOISE_MODEL,
    BaseNoiseModel,
)


class Noiseless(BaseNoiseModel):
    """Noiseless model for a simulated quantum circuit.

    Attributes
    ----------
    __type : NOISE_MODEL.NOISELESS
        Noiseless noise model type

    """

    def __init__(self):
        """Initialise Noiseless."""
        super().__init__(NOISE_MODEL.NOISELESS)
