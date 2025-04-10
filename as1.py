import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import resize_image, save_image

# === Configuration ===
index = 5
method = 'roberts'  # Options: roberts, compass, zero_cross, sobel, canny
save_filtered_image = True
save_plot_figure = True
resize = False
max_size = 300
assignment_number = 1

# === Helper: Auto-Detect Image Extension ===
def find_image_path(index, search_dir="images"):
    valid_exts = [".jpeg", ".jpg", ".png", ".bmp"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(script_dir, search_dir)
    for ext in valid_exts:
        path = os.path.join(images_dir, f"image{index}{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No valid image found for image{index} in {images_dir}")

# === Edge Detection Methods ===
def roberts_cross(img_gray):
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    gx = cv2.filter2D(img_gray, -1, kernel_x)
    gy = cv2.filter2D(img_gray, -1, kernel_y)
    return np.sqrt(gx**2 + gy**2)

def kirsch_compass(img_gray):
    g = [[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]
    kernels = [np.rot90(g, i) for i in range(8)]
    edge_maps = [cv2.filter2D(img_gray, -1, np.array(k)) for k in kernels]
    return np.max(edge_maps, axis=0)

def zero_crossing(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    zero_cross = np.zeros_like(laplacian)
    for y in range(1, laplacian.shape[0] - 1):
        for x in range(1, laplacian.shape[1] - 1):
            patch = laplacian[y-1:y+2, x-1:x+2]
            if np.max(patch) > 0 and np.min(patch) < 0:
                zero_cross[y, x] = 255
    return zero_cross

def sobel_edge(img_gray):
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return cv2.convertScaleAbs(magnitude)

def canny_edge(img_gray):
    return cv2.Canny(img_gray, 100, 200)

# === Image Loading & Edge Application ===
def load_image(path, resize=False, max_size=300):
    img = cv2.imread(path)
    if resize:
        img = resize_image(img, max_size)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_rgb, gray

def get_edge_map(gray, method):
    if method == 'roberts':
        return roberts_cross(gray)
    elif method == 'compass':
        return kirsch_compass(gray)
    elif method == 'zero_cross':
        return zero_crossing(gray)
    elif method == 'sobel':
        return sobel_edge(gray)
    elif method == 'canny':
        return canny_edge(gray)
    else:
        raise ValueError(f"Unsupported method: {method}")

# === Plotting ===
def plot_edge_result(image_rgb, edge_img, plot_save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(edge_img, cmap='gray')
    axes[1].set_title(f"{method.capitalize()} Edge Map")
    axes[1].axis('off')

    plt.tight_layout()

    if plot_save_path:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"[✔] Plot saved: {plot_save_path}")

    plt.show()

# === Run Everything ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = find_image_path(index)
    image_output_name = f'result_{method}_image{index}.jpeg'
    plot_output_name = f'plot_{method}_image{index}.jpeg'
    plot_output_path = os.path.join(script_dir, f"task{assignment_number}_plot_outputs", plot_output_name)

    # Load image and process
    _, image_rgb, gray = load_image(image_path, resize, max_size)
    result = get_edge_map(gray, method)

    if save_plot_figure:
        plot_edge_result(image_rgb, result, plot_output_path)
    else:
        plot_edge_result(image_rgb, result)

    if save_filtered_image:
        save_image(result, script_dir, image_output_name, assignment_number)
