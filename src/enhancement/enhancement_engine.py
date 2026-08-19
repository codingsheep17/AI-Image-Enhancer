import cv2
import numpy as np

def enhance_image(image, rules):
    #makin the copy of the img
    enhanced_img = image.copy()
    
    #read the enhancement rules
    resolution = rules["resolution_lvlr"]
    brightness = rules["brightness_lvlr"]
    contrast = rules["contrast_lvlr"]
    sharpness = rules["sharpness_lvlr"]
    noise = rules["noise_lvlr"]
    
    #noww the enhancement logic comes here
    
    #noise dealing (high case only)
    if noise == "high":
        enhanced_image = cv2.fastNlMeansDenoisingColored(
            enhanced_image,
            None,
            6,
            6,
            7,
            21
        )
        
    #brightness logic (dark case only)
    if brightness == "dark":
        gamma = 0.8

        lookup_table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ]).astype("uint8")

        enhanced_image = cv2.LUT(enhanced_image, lookup_table)
    
    #contrast enhancement (low only)
    """
    Converts BGR -> LAB.
    LAB separates:
    L: lightness
    A: green -> red information
    B: blue -> yellow information
    """
    if contrast == "low":
        lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    #now sharpening logic (blurry)
    if sharpness == "blurry":
        blurred = cv2.GaussianBlur(
            enhanced_image,
            (0, 0),
            2.0
        )
        enhanced_image = cv2.addWeighted(
            enhanced_image,
            1.5,
            blurred,
            -0.5,
            0
        )
    
    return enhanced_img