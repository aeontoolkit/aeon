"""Deep learning networks."""
__all__ = [
    "BaseDeepNetwork",
    "CNNNetwork",
    "EncoderNetwork",
    "FCNNetwork",
    "InceptionNetwork",
    "MLPNetwork",
    "ResNetNetwork",
    "TapNetNetwork",
    "AEFCNNetwork",
    "AEResNetNetwork",
]
from aeon.networks.ae_resnet import AEResNetNetwork
from aeon.networks.ae_fcn import AEFCNNetwork
from aeon.networks.base import BaseDeepNetwork
from aeon.networks.cnn import CNNNetwork
from aeon.networks.encoder import EncoderNetwork
from aeon.networks.fcn import FCNNetwork
from aeon.networks.inception import InceptionNetwork
from aeon.networks.mlp import MLPNetwork
from aeon.networks.resnet import ResNetNetwork
from aeon.networks.tapnet import TapNetNetwork
