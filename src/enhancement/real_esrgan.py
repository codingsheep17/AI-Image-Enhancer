import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# remaking the architecture neural network for the model
model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=4
)

model_path = "D:/Desktop/AI IMG ENHANCER/models/RealESRGAN_x4plus.pth"
# Load the raw checkpoint data
checkpoint = torch.load(model_path, map_location="cpu")

# Extract the EMA weights directly from the 'params_ema' key
if "params_ema" in checkpoint:
    state_dict = checkpoint["params_ema"]
elif "params" in checkpoint:
    state_dict = checkpoint["params"]
elif "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Load the fixed state dict into your model
model.load_state_dict(state_dict, strict=True)
model = model.to("cpu")

# Inference Engine Building
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,  
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)
print("Real-ESRGAN model loaded successfully!")
