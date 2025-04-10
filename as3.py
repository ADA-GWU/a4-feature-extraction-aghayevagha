import numpy as np
import cv2
import os
import time
from skimage import exposure
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour

# === CONFIGURATION ===
index = 3
resize = True
max_size = 800
assignment_number = 3
save_result = True


def find_image(index, search_dir="images"):
    valid_exts = [".jpeg", ".jpg", ".png", ".bmp"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(script_dir, search_dir)
    for ext in valid_exts:
        path = os.path.join(base_path, f"image{index}{ext}")
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No image found for image{index} in {search_dir}")

def load_image(path, resize=False, max_size=500):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    if resize:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

# === Mouse Drawing ===
def draw_contour(event, x, y, flags, param):
    global pts, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pts = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        pts.append((x, y))
        cv2.line(img, pts[-2], pts[-1], (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pts.append((x, y))
        cv2.line(img, pts[-2], pts[-1], (0, 255, 0), 2)
        cv2.line(img, pts[-1], pts[0], (0, 255, 0), 2)

# === Active Contour Processing ===
def apply_active_contour(image, initial_pts):
    gray = rgb2gray(image)
    equalized = exposure.equalize_adapthist(gray)
    edges = cv2.Canny((equalized * 255).astype(np.uint8), 100, 200)
    smoothed = gaussian(edges, 1)
    pts_np = np.fliplr(np.array(initial_pts))  # flip (x, y) -> (row, col)
    snake = active_contour(
        smoothed,
        pts_np,
        alpha=0.01,
        beta=1,
        gamma=0.01,
        convergence=0.01,
        max_num_iter=5000,
        boundary_condition='periodic'
    )
    return snake

# === Main ===
if __name__ == "__main__":
    img_path = find_image(index)
    img = load_image(img_path, resize, max_size)
    original_img = img.copy()

    global pts, drawing
    pts = []
    drawing = False

    cv2.namedWindow("press SPACE to apply active contour, R to reset")
    cv2.setMouseCallback("press SPACE to apply active contour, R to reset", draw_contour)

    print("Draw the initial contour, press SPACE to apply active contour, R to reset, or close the window to exit.")

    while True:
        cv2.imshow("press SPACE to apply active contour, R to reset", img)
        if cv2.getWindowProperty("press SPACE to apply active contour, R to reset", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting.")
            cv2.destroyAllWindows()
            exit()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            img = original_img.copy()
            pts = []
        elif key == 32:  # SPACE
            break

    cv2.destroyAllWindows()

    # Check if points are selected
    if not pts:
        print("[!] No contour points selected. Exiting.")
        exit()

    print("\n[i] Applying active contour... (this may take ~40s)")
    snake = apply_active_contour(original_img, pts)
    snake_pts = np.fliplr(snake).astype(np.int32)

    # Animate
    steps = 700
    for i in range(1, steps + 1):
        interp = pts + (snake_pts - pts) * i / steps
        temp = original_img.copy()
        cv2.polylines(temp, [interp.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.imshow("Contour Evolution", temp)
        if cv2.getWindowProperty("Contour Evolution", cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(20)

    cv2.destroyAllWindows()
    cv2.polylines(original_img, [snake_pts], True, (0, 255, 0), 3)
    cv2.imshow("Final Result", original_img)

    # Wait until user presses a key or closes the window
    while True:
        if cv2.getWindowProperty("Final Result", cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) != -1:
            break

    cv2.destroyAllWindows()

    if save_result:
        output_dir = os.path.join(os.path.dirname(__file__), f"task{assignment_number}_active_contour")
        os.makedirs(output_dir, exist_ok=True)
        name = f"image{index}_active_contour_result.png"
        result_path = os.path.join(output_dir, name)
        cv2.imwrite(result_path, original_img)
        print(f"Result saved to: {result_path}")
