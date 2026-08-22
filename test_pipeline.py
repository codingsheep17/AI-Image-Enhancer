import cv2

from src.preprocessing.image_preprocessor import analyze_image
from src.enhancement.enhancement_rules import determine_enhancement_rules
from src.enhancement.enhancement_engine import enhance_image


# Load image
image = cv2.imread(
    "D:/Desktop/AI IMG ENHANCER/real_esrgan_output.png"
)

# 1. Analyze image
analysis = analyze_image(image)

print("\n=== ANALYSIS ===")
print(analysis)


# 2. Generate enhancement rules
rules = determine_enhancement_rules(analysis)

print("\n=== ENHANCEMENT RULES ===")
print(rules)


# 3. Apply enhancement rules
enhanced_image = enhance_image(image, rules)


# 4. Save result
cv2.imwrite(
    "D:/Desktop/AI IMG ENHANCER/integration_test_output.png",
    enhanced_image
)

print("\n=== PIPELINE ===")
print("Analysis → Rules → Enhancement")
print("Integration test completed successfully.")

