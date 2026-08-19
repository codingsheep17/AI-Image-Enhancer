# rule based system connected with the img preprocessing (analyze image method)
def determine_enhancement_rules(analysis):
    
    #working on the resolution rules fixation
    width = analysis['width']
    height = analysis['height']
    
    # Resolution classification
    if width < 256 or height < 256:
        resolution = "very_low"
    elif width < 720 or height < 720:
        resolution = "low"
    elif width >= 1080 and height >= 1080:
        resolution = "high"
    else:
        resolution = "acceptable"


    #working on the brightness rules
    brightness = analysis['brightness']
    if brightness < 60:
        brightness_level = "dark"
    elif brightness < 180:
        brightness_level = "normal"
    elif brightness < 220:
        brightness_level = "bright"
    else:
        brightness_level = "very_bright"
        
    #now working on the contrast rule
    contrast = analysis['contrast']
    if contrast < 30:
        contrast_level = "low"
    elif contrast < 70:
        contrast_level = "normal"
    else:
        contrast_level = "high"

    #now time for the sharpness to determine blurr
    sharpness = analysis['sharpness']
    if sharpness < 100:
        sharpness_level = "blurry"
    elif sharpness < 500:
        sharpness_level = "normal"
    else:
        sharpness_level = "sharp"
        
    #now setting rules for the noise
    noise = analysis['noise_level']
    if noise < 5:
        noise_level = "low"
    elif noise < 15:
        noise_level = "moderate"
    else:
        noise_level = "high"
        
    return {
        "resolution_lvlr": resolution,
        "brightness_lvlr":brightness_level,
        "contrast_lvlr":contrast_level,
        "sharpness_lvlr":sharpness_level,
        "noise_lvlr":noise_level
    }
    