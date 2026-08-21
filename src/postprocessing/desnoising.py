import cv2

def denoise_image(image):
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        5,
        5,
        7,
        21
    )

    return denoised

if __name__ == "__main__":
    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png")

    if image is None:
        raise ValueError("Unable to load image.")

    denoised = denoise_image(image)

    cv2.imwrite("denoised_test.png", denoised)

    print("Denoising completed")