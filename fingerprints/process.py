import cv2

# Fingerprint image ka path
image_path = "fingerprints/fingerprint_960x300.png"

# Image read karo
image = cv2.imread(image_path)

# Check karo image properly load hui ya nahi
if image is None:
    print("❌ Fingerprint image load nahi hui!")
else:
    print("✅ Fingerprint image successfully loaded!")

    # Grayscale mein convert
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Image width:", gray.shape[1])
    print("Image height:", gray.shape[0])

    # Grayscale image save karo
    cv2.imwrite("fingerprints/fingerprint_gray.png", gray)
# ==========================================
# FINGERPRINT ENHANCEMENT
# ==========================================

# Contrast improve karna
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

# Thoda noise remove karna
enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

# Enhanced fingerprint save karo
cv2.imwrite(
    "fingerprints/fingerprint_enhanced.png",
    enhanced
)

print("✅ Fingerprint enhancement complete!")
print("✅ Enhanced fingerprint saved!")
# ==========================================
# FINGERPRINT RIDGE EXTRACTION
# ==========================================

# Fingerprint ridges ko black/white mein separate karna
_, ridge_image = cv2.threshold(
    enhanced,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Small noise remove karna
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

ridge_image = cv2.morphologyEx(
    ridge_image,
    cv2.MORPH_OPEN,
    kernel
)

# Ridge image save karo
cv2.imwrite(
    "fingerprints/fingerprint_ridges.png",
    ridge_image
)

print("✅ Fingerprint ridge extraction complete!")
print("✅ Ridge image saved!")
# ==========================================
# FINGERPRINT FEATURE EXTRACTION
# ==========================================

# Ridge image se useful statistical features
mean_value = ridge_image.mean()
white_pixels = cv2.countNonZero(ridge_image)
total_pixels = ridge_image.shape[0] * ridge_image.shape[1]
ridge_density = white_pixels / total_pixels

print("================================")
print("FINGERPRINT FEATURES")
print("================================")
print("Mean intensity:", mean_value)
print("Ridge pixels:", white_pixels)
print("Total pixels:", total_pixels)
print("Ridge density:", ridge_density)

print("✅ Feature extraction complete!")