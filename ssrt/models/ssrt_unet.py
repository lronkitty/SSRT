### 任意大小。x编码.nomask
from tkinter import X
try:
    from hydra.utils import to_absolute_path
except:
    print("Hydra not found, using relative paths")
    pass
import logging

import torch
import torch.nn as nn

from .base import BaseModel
import ssrt.models.layers as layers
# from ssrt.models.layers.combinations import *
# from ssrt.models.layers.brt_modules import BlockRecurrentAttention
# from ssrt.models.layers.network_swinir import *
from ssrt.models.layers.ssrt import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ssrt_Unet(BaseModel):
    def __init__(self,base,channels,
        ssl=0,
        n_ssl=0,ckpt=None,):
        super().__init__(**base)
        self.channels = channels
        self.layers_params = layers
        self.ssl = ssl
        self.n_ssl = n_ssl
        logger.debug(f"ssl : {self.ssl}, n_ssl : {self.n_ssl}")

        # self.init_layers()
        
        upscale = 1
        window_size = 8
        height = 64
        width = 64
        self.net = ssrt(upscale=1, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[2,2,6,2,2],
                   embed_dim=self.channels, num_heads=[2,2,2,2,2], mlp_ratio=2, upsampler=None,in_chans=1,gate='sru',if_mlp_s=True)
        logger.info(f"Using SSL : {self.ssl}")
        self.ckpt = ckpt
        if self.ckpt is not None:
            try:
                logger.info(f"Loading ckpt {self.ckpt!r}")
                d = torch.load(to_absolute_path(self.ckpt))
                self.load_state_dict(d["state_dict"])
            except:
                print("Could not load ckpt")
                pass

    def forward(self,x, mode=None, img_id=None, sigmas=None, ssl_idx=None, **kwargs
    ):
        x = self.net(x)
        return x
