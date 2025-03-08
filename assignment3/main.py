import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the image
input_image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
input_image = input_image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Compute Discrete Cosine Transform (DCT)
dct_image = cv2.dct(input_image)

# Haar Wavelet Transform (Manually Implemented)
def haar_wavelet_transform(image):
    # Perform a single-level Haar DWT
    rows, cols = image.shape
    LL = np.zeros((rows // 2, cols // 2))
    LH = np.zeros((rows // 2, cols // 2))
    HL = np.zeros((rows // 2, cols // 2))
    HH = np.zeros((rows // 2, cols // 2))

    # Apply Haar DWT on rows
    for i in range(rows):
        for j in range(0, cols, 2):
            avg = (image[i, j] + image[i, j + 1]) / 2
            diff = (image[i, j] - image[i, j + 1]) / 2
            LL[i, j // 2] = avg
            LH[i, j // 2] = diff

    # Apply Haar DWT on columns
    for j in range(cols):
        for i in range(0, rows, 2):
            avg = (LL[i, j] + LL[i + 1, j]) / 2
            diff = (LL[i, j] - LL[i + 1, j]) / 2
            LL[i // 2, j] = avg
            HL[i // 2, j] = diff

    HH = LH + HL  # Combine LH and HL for the high-frequency components
    return LL, LH, HL, HH

# Compute Discrete Wavelet Transform (DWT) using the Haar wavelet
LL, LH, HL, HH = haar_wavelet_transform(input_image)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.log(np.abs(dct_image) + 1), cmap='gray')
plt.title('DCT of Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(LL, cmap='gray')
plt.title('Low Frequency (LL) - DWT')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.log(np.abs(HH) + 1), cmap='gray')
plt.title('High Frequency (HH) - DWT')
plt.axis('off')

# Save output images
cv2.imwrite('dct_image.jpg', cv2.normalize(dct_image, None, 0, 255, cv2.NORM_MINMAX))
cv2.imwrite('ll_wavelet.jpg', cv2.normalize(LL, None, 0, 255, cv2.NORM_MINMAX))
cv2.imwrite('hh_wavelet.jpg', cv2.normalize(HH, None, 0, 255, cv2.NORM_MINMAX))

# Show the plots
plt.tight_layout()
plt.show()
