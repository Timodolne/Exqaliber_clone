from circuit_models.noise_model.base import NOISE_MODEL, BaseNoiseModel

class Noiseless(BaseNoiseModel):
    '''Noiseless model'''

    def __init__(self):
        super().__init__(NOISE_MODEL.NOISELESS)
