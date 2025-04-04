import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def compute_dct(image):
    image_float = np.float32(image) / 255.0  # Normalize to 0-1
    dct_rows = cv2.dct(image_float)  # Apply DCT on rows
    dct = cv2.dct(dct_rows.T).T  # Apply DCT on columns

    # Normalizing DCT with log scaling
    dct_vis = np.abs(dct)
    dct_vis = np.log1p(dct_vis)
    dct_vis = cv2.normalize(dct_vis, None, 0, 255, cv2.NORM_MINMAX)

    return dct, dct_vis  # Return both raw and normalized DCT


def compute_dwt(image):
    # Apply 2D Discrete Wavelet Transform
    coeffs2 = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs2
    return cA, cH, cV, cD


def visualize_results(image, dct, dct_vis, cA, cH, cV, cD):
    plt.figure(figsize=(12, 12))

    # Original Image
    plt.subplot(3, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

    # DCT
    plt.subplot(3, 2, 3), plt.imshow(dct, cmap='gray'), plt.title('DCT Spectrum (Raw)')
    plt.subplot(3, 2, 4), plt.imshow(dct_vis, cmap='gray'), plt.title('DCT Spectrum (Normalized)')

    # DWT
    plt.subplot(3, 4, 9), plt.imshow(cA, cmap='gray'), plt.title('DWT Approximation')
    plt.subplot(3, 4, 10), plt.imshow(cH, cmap='gray'), plt.title('DWT Horizontal')
    plt.subplot(3, 4, 11), plt.imshow(cV, cmap='gray'), plt.title('DWT Vertical')
    plt.subplot(3, 4, 12), plt.imshow(cD, cmap='gray'), plt.title('DWT Diagonal')

    plt.tight_layout()
    plt.show()

def main():
    image_path = 'cameraman.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    dct, dct_vis = compute_dct(image)
    cA, cH, cV, cD = compute_dwt(image)

    visualize_results(image, dct, dct_vis, cA, cH, cV, cD)


if __name__ == "__main__":
    main()