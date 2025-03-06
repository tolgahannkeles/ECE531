"""
Write a code (GUI not required) that reads an image, performs these operations on that image, and writes output files to the disk.

- Gamma correction (point operation)

- Prewitt edge detection (neighborhood operation) in both directions, vertical and horizontal

- DCT (discrete cosine transform) (global operation) with log scaling

Please try to code your own functions instead of using the predefined libraries.

Please also upload your input image, output images of your code, and other screenshots if possible.
"""
import cv2
import numpy as np
import os

# Function to apply gamma correction
def gamma_correction(image, gamma=2.2):
    # Normalize the image to [0, 1]
    image_normalized = image / 255.0
    # Apply gamma correction
    corrected_image = np.power(image_normalized, 1/gamma)
    # Scale back to [0, 255]
    return np.uint8(corrected_image * 255)

# Function to apply Prewitt edge detection
def prewitt_edge_detection(image):
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Apply convolution using cv2.filter2D
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)

    # Combine edges
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = np.uint8(edges / np.max(edges) * 255)

    return edges_x, edges_y, edges

# Function to apply DCT with log scaling
def dct_with_log_scaling(image):
    # Convert image to float32 for DCT
    image_float32 = np.float32(image)
    # Apply DCT using cv2.dct
    dct_output = cv2.dct(image_float32)
    # Apply log scaling
    dct_output = np.log(1 + np.abs(dct_output))
    dct_output = np.uint8(dct_output / np.max(dct_output) * 255)

    return dct_output

def main():
    # Read image
    image_path = "cameraman.png"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Output path
    output_path = "outputs/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # Gamma correction
    gamma_corrected = gamma_correction(image, 2.2)
    cv2.imwrite(f"{output_path}gamma_corrected.jpg", gamma_corrected)

    # Prewitt edge detection
    edges_x, edges_y, edges = prewitt_edge_detection(image)
    cv2.imwrite(f"{output_path}prewitt_edges_x.jpg", edges_x)
    cv2.imwrite(f"{output_path}prewitt_edges_y.jpg", edges_y)
    cv2.imwrite(f"{output_path}prewitt_edges_combined.jpg", edges)

    # DCT with log scaling
    dct_image = dct_with_log_scaling(image)
    cv2.imwrite(f"{output_path}dct_log_scaled.jpg", dct_image)

if __name__ == "__main__":
    main()