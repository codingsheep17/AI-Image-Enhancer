import cv2

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)

    sharpened = cv2.addWeighted(
        image,
        1.8,
        blurred,
        -0.8,
        0
    )
#did made changes inside the function 

    return sharpened

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(gray, cv2.CV_64F).var()

#added the new testing function
if __name__ == "__main__":
        original = cv2.imread("D:/Desktop/AI IMG ENHANCER/test1.png")
        esrgan = cv2.imread("D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png")
        sharpened = cv2.imread("D:/Desktop/AI IMG ENHANCER/sharpened_test.png")

        print("Original:", calculate_sharpness(original))
        print("Real-ESRGAN:", calculate_sharpness(esrgan))
        print("Sharpened:", calculate_sharpness(sharpened))
        
#Original - Blur -> Edge emphasis -> Sharper image
