import cv2
import os

image_path = ''
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_image(image_path):
    #checking if the path of the image contains an img actually
    if not os.path.exists(image_path):
        return False, "Image file does not exist."
    
    #checking if the size of the file is ok or not 
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        return False, "Image file is empty."
    
    #getting the extensions
    extension = os.path.splitext(image_path)[1].lower()
    
    #checking the file extension
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported image format: {extension}"
    
    
    #now the opencv part begins
    image = cv2.imread(image_path) #it loads the image in the form of BGR (blue, green, red)
    
    #checking once again
    if image is None:
        return False, "(2nd check) Unable to read image."

    
    #understanding the image and normalizing image
    """Checking What is the color channel of image"""
    
    #channel setting to 3
    if len(image.shape) == 3:
        channels = image.shape[2]
        #if the channels = 4 so means that it's a BGRA (converting to BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)   
    else:
        channels = 1
        #if the channels = 1 so means that it's a grayscale (converting the grayscale to BGR)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image,{
        "channel":channels
    }
    
image, channel = validate_image(image_path)
    
def analyze_image(image):
    #inspecting the pixel value of the image
    image_type = image.dtype
    image_min_pixel = image.min()
    image_max_pixel = image.max()
            
    #checking the blue, green and red pixel values 
    pixel = image[0, 0]
        
    #checking the height, width of the image
    height, width = image.shape[:2]
        
    #aspect ratio
    aspect_ratio = width/height
    
    #calculating the image brightness too (by converting into grayscale image)
    gray_scaled_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    """Calculating the mean of the image for brightness"""
    #Brightness =  sum of all pixel intensities/no of pixels
    average_brightness = gray_scaled_img.mean()
    
    #calculating the contrast of the image
    contrast = gray_scaled_img.std()
    
    #now calculating the sharpness / blur detection (using the laplacian variance)
    laplacian = cv2.Laplacian(gray_scaled_img, cv2.CV_64F)
    sharpness = laplacian.var()
    
    #noise esmtimation (the diff b/w image and blurred version of itself)
    blurred = cv2.GaussianBlur(gray_scaled_img, (3,3), 0)
    noise = gray_scaled_img.astype('float32') = blurred.astype('float32')
    #converted into float32 as sometimes it can produce -ve value
    noise_lvl = noise.std()
    #low noise -> clean, high noise -> noisy
    
    return {
        "height":height,
        "width":width,
        "image_type":image_type,
        "min_pixel":image_min_pixel,
        "max_pixel":image_max_pixel,
        "pixel":pixel,
        "aspect_ratio":aspect_ratio,
        "brightness":average_brightness,
        "contrast":contrast,
        "sharpness":sharpness,
        "noise_level":noise_lvl
        }
    
    #these will be conncted to the enhancement_rules.py for the creation of the ruling system
    
if __name__ == "__main__":
    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png")

    analysis = analyze_image(image)

    print(analysis)