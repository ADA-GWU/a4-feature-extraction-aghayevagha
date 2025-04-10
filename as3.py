import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.interpolate import splprep, splev
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from utils import save_image  



# === SETTINGS ===
index = 3  # Just change this number to select another image

# resize if you want
resize = True
max_size = 800

#blur before processing, increases the performance
blur = True

# save the figure.
save_filtered_image = True
save_plot_figure = True

points = []


# assignment 
assignment =3

# some images are in different format, it automatically finds the correct path 
def find_image_path(index, search_dir="images"):
    valid_exts = [".jpeg", ".jpg", ".png", ".bmp"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir = os.path.join(script_dir, search_dir)
    for ext in valid_exts:
        path = os.path.join(images_dir, f"image{index}{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No valid image found for image{index} in {images_dir}")

# loading image and resizing if it is too big
def load_image(image_path, resize=True, max_size=500):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    if resize:
        h, w = image_bgr.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return image_bgr, image_rgb, gray

# point selection
def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((y, x))  # (row, col)

# i ordered the points if the user randomly selects the points
# I apply smoothing as it increased the performance
def smooth_initial_curve(points, num_points=200):
    pts = np.array(points)
    if len(pts) < 3:
        return pts
    hull = ConvexHull(pts)
    ordered = pts[hull.vertices]
    ordered = np.vstack([ordered, ordered[0]])  # Close loop
    tck, _ = splprep([ordered[:, 0], ordered[:, 1]], s=0, per=True)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    return np.stack([x_new, y_new], axis=1)

# appling snake algorithm
def apply_active_contour(gray_image, init_curve):
    img_norm = gray_image.astype(np.float64) / 255.0
    smoothed = gaussian(img_norm, sigma=3)
    snake = active_contour(
        smoothed,
        init_curve,
        alpha=0.01,
        beta=4.0,
        gamma=0.01,
        max_num_iter=2500,
        convergence=0.1
    )
    return snake

def plot_results(image_rgb, snake, save_path=None):
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=2, label='Snake')
    ax.plot([p[1] for p in points], [p[0] for p in points], 'ro', markersize=3, label='Initial Points')
    ax.axis("off")
    ax.set_title("Active Contour Result")
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Auto-find image from provided images, or change the image path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = find_image_path(index)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_bgr, image_rgb, gray = load_image(image_path, resize, max_size)

    # you also need to change the result path if you pick your own image from local.
    image_output_name = f'result_snake_{image_name}.jpeg'
    plot_output_name = f'plot_snake_{image_name}.jpeg'
    plot_output_path = os.path.join(script_dir, "task3_plot_outputs", plot_output_name)

    # Point selection window
    cv2.namedWindow("Select Points")
    cv2.setMouseCallback("Select Points", select_points)

    while True:
        img_copy = image_rgb.copy()
        for point in points:
            cv2.circle(img_copy, (point[1], point[0]), 3, (255, 0, 0), -1)
        cv2.imshow("Select Points", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == 13:  # Enter key
            print("Points selected!")
            break

    cv2.destroyAllWindows()

    # Check minimum number of points
    if len(points) < 3:
        print(" Need at least 3 points to run active contour. Try again.")
        exit(1)

    # Run snake
    init_points = smooth_initial_curve(points)
    snake_contour = apply_active_contour(gray, init_points)

    # Plot and save results
    if save_plot_figure:
        plot_results(image_rgb, snake_contour, plot_output_path)
    else:
        plot_results(image_rgb, snake_contour)

    if save_filtered_image:
        save_image(image_rgb, script_dir, image_output_name, assignment)
