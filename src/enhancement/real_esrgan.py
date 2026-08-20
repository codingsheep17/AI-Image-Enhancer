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

# Load the fixed state dict in model
model.load_state_dict(state_dict, strict=True)
model = model.to("cpu")

# Inference Engine Building
upsampler = RealESRGANer(
    scale=4,
    model_path=model_path,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=False
)

#enhancer function
def enhance_image(image):
    output, _ = upsampler.enhance(image, outscale=4)
    return output

#testing the img enhancement
if __name__ == "__main__":
    import cv2

    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/test1.png")

    if image is None:
        raise ValueError("Unable to load test image.")

    enhanced_image = enhance_image(image)

    cv2.imwrite("real_esrgan_output.png", enhanced_image)

    print("Image enhanced successfully!")

# first img enchancing test passed (although not so satisfying)