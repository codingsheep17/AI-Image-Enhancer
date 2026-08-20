import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

#remaking the architecture neural network for the model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

#load state dict is adding the weights into our RRDBNet
model_path = "D:/Desktop/AI IMG ENHANCER/models/RealESRGAN_x4plus.pth"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model = model.to("cpu")

