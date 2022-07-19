from circuit_models.noise_model.base import NOISE_MODEL, BaseNoiseModel

class Noiseless(BaseNoiseModel):
    '''Noiseless model
    
    Attributes
    ----------
    __type : NOISE_MODEL.NOISELESS
        Noiseless noise model type
    '''

    def __init__(self):
        super().__init__(NOISE_MODEL.NOISELESS)
