import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import resize_image, save_image

# === Configuration ===
index = 2  # Change to the desired image index

#play with parameters for different images, uncommentd the corresponding for best result as different bicycles have different
#radiuses of tires
#kernel, min_rad, max_rad = 7,190,200  # for image 1
kernel, min_rad, max_rad = 5,40,60     # for image 2

save_filtered_image = True
save_plot_figure = True
resize = False
max_size = 300
blur = True
assignment_number = 2

# === Find the Image Path ===
def find_image_path(index, search_dir="images"):
    valid_exts = [".jpeg", ".jpg", ".png", ".bmp"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(script_dir, search_dir)
    for ext in valid_exts:
        path = os.path.join(images_dir, f"image{index}{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No valid image found for image{index} in {images_dir}")

# === Pipeline Functions ===
def load_image(path, resize=False, max_size=300):
    image = cv2.imread(path)
    if resize:
        image = resize_image(image, max_size)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, image_rgb, gray

def detect_edges(gray_img, use_blur=False, kernel_size=kernel, canny_threshold1=50, canny_threshold2=150):
    if use_blur:
        gray_img = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 0)
    return cv2.Canny(gray_img, canny_threshold1, canny_threshold2)

def detect_lines(image_rgb, edges):
    line_img = image_rgb.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return line_img

def detect_circles(gray_img, image_rgb):
    circle_img = image_rgb.copy()
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=40, minRadius=min_rad, maxRadius=max_rad)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            cv2.circle(circle_img, (x, y), r, (255, 0, 0), 2)
            cv2.circle(circle_img, (x, y), 2, (0, 0, 255), 3)
    return circle_img

def plot_results(image_rgb, edges, result_img, plot_save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Original Image', 'Canny Edges', 'Hough Lines & Circles']
    images = [image_rgb, edges, result_img]

    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        rect = patches.Rectangle((0, 0), img.shape[1], img.shape[0],
                                 linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    if plot_save_path:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved: {plot_save_path}")
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = find_image_path(index)

    image_output_name = f'result_hough_image{index}.jpeg'
    plot_output_name = f'plot_hough_image{index}.jpeg'
    plot_output_path = os.path.join(script_dir, f"task{assignment_number}_plot_outputs", plot_output_name)

    # === Pipeline ===
    image_bgr, image_rgb, gray = load_image(image_path, resize, max_size)
    edges = detect_edges(gray, use_blur=blur)
    with_lines = detect_lines(image_rgb, edges)
    final_result = detect_circles(gray, with_lines)

    if save_plot_figure:
        plot_results(image_rgb, edges, final_result, plot_output_path)
    else:
        plot_results(image_rgb, edges, final_result)

    if save_filtered_image:
        save_image(final_result, script_dir, image_output_name, assignment_number)
