import cv2

def sharpen_image(image):
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    return sharpened

if __name__ == "__main__":
    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png")

    sharpened = sharpen_image(image)

    cv2.imwrite("sharpened_test.png", sharpened)

    print("Sharpening completed!")
#Original - Blur -> Edge emphasis -> Sharper image
