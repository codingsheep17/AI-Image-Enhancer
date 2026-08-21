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

