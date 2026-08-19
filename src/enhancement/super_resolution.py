from cv2 import dnn_superres

#importing the dnn method from the cv2 for the model
sr = dnn_superres.DnnSuperResImpl.create()
sr.readModel("D:/Desktop/AI IMG ENHANCER/models/FSRCNN_x4.pb")
sr.setModel("fsrcnn", 4)

def upscale_image(image):
    enhanced_image = sr.upsample(image)
    return enhanced_image


#for the testing purposes 
if __name__ == "__main__":
    import cv2
    image = cv2.imread("D:/Desktop/AI IMG ENHANCER/test1.png")

    if image is None:
        raise ValueError("Unable to load test image.")

    enhanced_image = upscale_image(image)

    cv2.imwrite("test_upscaled.jpg", enhanced_image)