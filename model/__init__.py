from .base_model import BaseModel
from .ULMEGAN import ULMEGANModel



def create_model(opt):
    # specify model name here
    instance = ULMEGANModel()
    instance.initialize(opt)
    instance.setup()
    return instance

