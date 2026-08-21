import cv2

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    enhanced_l = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_image

if __name__ == "__main__":
    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png")

    contrast = enhance_contrast(image)

    cv2.imwrite("contrast_test.png", contrast)

    print("Contrast Testing Completed!")