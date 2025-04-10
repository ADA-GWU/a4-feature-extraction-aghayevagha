import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#if you want to use your own images, go to the main part and change image names, also you can change output names too.
 # ->/pair{index}/ change for different pais
pair_index = 3    

# Lowe’s match filter threshold, you can adjust it 
MIN_MATCH_COUNT = 5  

# Resize large images if u wish
resize = False             
max_size = 600            

 #  Set this to True to save output files
save_results = True     

# Used in result folder naming
assignment = 4            

# gets image
def find_image_in_pair(pair_idx, name="image1", search_dir="images/pairs"):
    valid_exts = [".jpeg", ".jpg", ".png", ".bmp"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.join(script_dir, search_dir, f"pair{pair_idx}")
    for ext in valid_exts:
        img_path = os.path.join(base_path, f"{name}{ext}")
        if os.path.isfile(img_path):
            return img_path
    raise FileNotFoundError(f"{name} not found in pair{pair_idx}")

def load_image(path, resize=True, max_size=600):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to load: {path}")
    if resize:
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
    if blur:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

def save_image(img, name, subfolder="orb_outputs"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(script_dir, f"task{assignment}_{subfolder}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, name)
    cv2.imwrite(out_path, img)
    print(f"Image saved to: {out_path}")
    return out_path

def save_plot(fig, name, subfolder="orb_outputs"):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    out_dir = os.path.join(script_dir, f"task{assignment}_{subfolder}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved to: {path}")
    return path

# === Matching Logic ===
def match_images(img1, img2):
    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("Descriptors missing in one or both images.")
        return None, None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < MIN_MATCH_COUNT:
        print(f"Not enough good matches: {len(good)}")
        return None, None, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None or mask is None or np.sum(mask) < MIN_MATCH_COUNT:
        print("Homography failed or too few inliers.")
        return None, None, None, None

    h, w = img1.shape[:2]
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, M)

    img2_boxed = img2.copy()
    cv2.polylines(img2_boxed, [np.int32(projected)], True, (0,255,0), 3)

    match_vis = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return match_vis, img2_boxed, len(good), good


if __name__ == "__main__":
    blur = False  
    img1_path = find_image_in_pair(pair_index, "image1")
    img2_path = find_image_in_pair(pair_index, "image2")

    img1 = load_image(img1_path, resize=resize, max_size=max_size)
    img2 = load_image(img2_path, resize=resize, max_size=max_size)

    match_img, boxed_img, match_count, matches = match_images(img1, img2)

    if match_img is not None:
        print(f"Found {match_count} good matches.")

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))
        axs[0].imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("ORB + BRIEF Matches")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Projected Object in Image 2")
        axs[1].axis("off")
        plt.tight_layout()

        if save_results:
            base1 = os.path.basename(img1_path)
            base2 = os.path.basename(img2_path)
            match = f"match_{base1}_{base2}.jpg"
            box = f"box_{base1}_{base2}.jpg"
            plot = f"plot_{base1}_{base2}.jpg"
            save_image(match_img, match)
            save_image(boxed_img, box)
            save_plot(fig, plot)
        else:
            plt.show()
    else:
        print("Match or homography failed.")
