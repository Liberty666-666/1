import cv2
import mediapipe as mp
import numpy as np
import os

def crop_face_regions(image_path, output_dir):
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        min_detection_confidence=0.5
    )

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Detect facial landmarks
    results = face_mesh.process(image_rgb)
    if not results.multi_face_landmarks:
        print("No face detected.")
        face_mesh.close()
        return {}

    landmarks = results.multi_face_landmarks[0].landmark

    # Helper: get landmark coordinates
    def get_landmark_point(landmark_idx):
        lm = landmarks[landmark_idx]
        return (int(lm.x * width), int(lm.y * height))

    # Define five regions
    regions = {
        "left_forehead": [10, 109, 67, 103, 54, 21],
        "right_forehead": [10, 338, 297, 332, 284, 251],
        "nose_upper_cheeks": [1, 2, 98, 327, 50, 280],
        "left_lower_cheek_chin": [50, 101, 36, 206, 207, 187, 152],
        "right_lower_cheek_chin": [280, 330, 266, 426, 427, 411, 152],
    }

    # Store cropped image paths
    cropped_images = {}

    # Create mask and crop each region
    for region_name, points in regions.items():
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([get_landmark_point(idx) for idx in points], dtype=np.int32)
        if len(pts) > 0:
            cv2.fillPoly(mask, [pts], 255)

        coords = np.where(mask == 255)
        if len(coords[0]) > 0:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            cropped_image = image[y_min:y_max+1, x_min:x_max+1]
            output_path = os.path.join(output_dir, f"{region_name}.jpg")
            cv2.imwrite(output_path, cropped_image)
            cropped_images[region_name] = output_path
        else:
            print(f"Failed to crop {region_name}: No region detected")
            cropped_images[region_name] = None

    # Release resources
    face_mesh.close()
    return cropped_images

# Example usage:
image_path = "D:\MEC202group project\Segmentation\19.jpg"
output_dir = "D:\MEC202group project\Segmentation"
cropped_paths = crop_face_regions(image_path, output_dir)

print("Cropped image paths:", cropped_paths)
