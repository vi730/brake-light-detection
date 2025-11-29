"""
Dataset Preparation for Vehicle Brake Light Detection
Generates synthetic training dataset by overlaying vehicle images onto backgrounds
and creates YOLO format annotations
"""

import cv2
from PIL import Image
import numpy as np
import os
import random
import shutil
from glob import glob
from tqdm import tqdm

# ==================== Configuration ====================

OBJ_CLASSES = ['Normal', 'Braking']
CUR_DIR = os.curdir
DATA_SRC_PATH = os.path.join(CUR_DIR, 'data_source')
DATASET_PATH = os.path.join(CUR_DIR, 'dataset')
DEST_IMG_DIR = os.path.join(DATA_SRC_PATH, 'MergedImages')
LABEL_DIR = os.path.join(DATA_SRC_PATH, 'Labels')
DATASET_FOLDERS = ['train', 'valid', 'test']

IMGS_PER_CLASS = 500 # Recommended: 500 for training, 10 for testing
SPLIT_RATIO = [0.80, 0.10, 0.10]

# Image overlay parameters
BOTTOM_MARGIN = 20
SIDE_MARGIN = 50
ADJUSTMENT_FACTOR = 1.5
Y_RAND_FACTOR = 5
X_RAND_STEPS = 1
Y_RAND_STEPS = 1


# ==================== Directory Management ====================

def create_directory_structure():
    """Create directory structure for dataset"""
    try:
        os.makedirs(DATA_SRC_PATH, exist_ok=True)
        os.makedirs(DATASET_PATH, exist_ok=True)
        
        data_src_folders = OBJ_CLASSES + ['Backgrounds', 'MergedImages', 'Labels']
        for folder_name in data_src_folders:
            os.makedirs(os.path.join(DATA_SRC_PATH, folder_name), exist_ok=True)
        
        for folder_name in DATASET_FOLDERS:
            folder_path = os.path.join(DATASET_PATH, folder_name)
            os.makedirs(os.path.join(folder_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(folder_path, 'labels'), exist_ok=True)
        
    except Exception as e:
        print(f'Error creating directories: {e}')
        raise


# ==================== Image Processing ====================

def square_image(img):
    """Crop image to square from center"""
    h, w, c = img.shape
    diff = abs(w - h)
    
    if w >= h:
        x1 = diff // 2
        y1 = 0
        x2 = x1 + h
        y2 = h
    else:
        y1 = diff // 2
        x1 = 0
        y2 = y1 + w
        x2 = w
    
    return img[y1:y2, x1:x2]


def cv2pil(img):
    """Convert OpenCV to PIL format"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def pil2cv(img):
    """Convert PIL to OpenCV format"""
    img_array = np.array(img)
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


# ==================== Label Generation ====================

def calculate_position_and_label(src_img, dest_img):
    """Calculate optimal position for foreground and generate YOLO label"""
    src_h, src_w, _ = src_img.shape
    dest_h, dest_w, _ = dest_img.shape
    
    # Calculate Y position with depth perception
    max_src_y = (dest_h - src_h) - BOTTOM_MARGIN
    y_offset = round((100 - ((src_h / dest_h) * 100)) * ADJUSTMENT_FACTOR)
    src_y = max_src_y - y_offset
    src_y = random.randrange(
        max(0, src_y - Y_RAND_FACTOR),
        min(max_src_y, src_y + Y_RAND_FACTOR),
        Y_RAND_STEPS
    )
    
    # Calculate X position
    min_src_x = SIDE_MARGIN
    max_src_x = max(min_src_x + 10, dest_w - src_w - SIDE_MARGIN)
    src_x = random.randrange(min_src_x, max_src_x, X_RAND_STEPS)
    
    # YOLO format: center_x center_y width height (normalized)
    bbox_center_x = (src_x + src_w / 2) / dest_w
    bbox_center_y = (src_y + src_h / 2) / dest_h
    bbox_width = src_w / dest_w
    bbox_height = src_h / dest_h
    
    label_text = f"{bbox_center_x:.7f} {bbox_center_y:.7f} {bbox_width:.7f} {bbox_height:.7f}"
    
    return src_x, src_y, label_text


def overlay_images(src_img, dest_img):
    """Overlay foreground image onto background"""
    dest_copy = dest_img.copy()
    src_x, src_y, label_text = calculate_position_and_label(src_img, dest_img)
    
    src_pil = cv2pil(src_img)
    dest_pil = cv2pil(dest_copy)
    dest_pil.paste(src_pil, (src_x, src_y))
    
    merged_img = pil2cv(dest_pil)
    
    return merged_img, label_text


# ==================== Dataset Generation ====================

def generate_dataset():
    """Generate training dataset by overlaying vehicle images onto backgrounds"""
    bg_dir = os.path.join(DATA_SRC_PATH, 'Backgrounds')
    if not os.path.exists(bg_dir):
        raise FileNotFoundError(f"Background folder not found: {bg_dir}")
    
    bg_imgs_list = os.listdir(bg_dir)
    if not bg_imgs_list:
        raise ValueError(f"No images in background folder: {bg_dir}")
    
    for class_id, class_name in enumerate(OBJ_CLASSES):
        print(f"\nProcessing class: {class_name} (ID: {class_id})")
        
        fg_dir = os.path.join(DATA_SRC_PATH, class_name)
        if not os.path.exists(fg_dir):
            raise FileNotFoundError(f"Foreground folder not found: {fg_dir}")
        
        fg_imgs_list = os.listdir(fg_dir)
        if not fg_imgs_list:
            raise ValueError(f"No images in foreground folder: {fg_dir}")
        
        for n in tqdm(range(IMGS_PER_CLASS), desc=f"Generating {class_name}"):
            fg_img_file = os.path.join(fg_dir, random.choice(fg_imgs_list))
            bg_img_file = os.path.join(bg_dir, random.choice(bg_imgs_list))
            
            fg_img = cv2.imread(fg_img_file)
            bg_img = cv2.imread(bg_img_file)
            
            if fg_img is None:
                raise IOError(f"Cannot read foreground image: {fg_img_file}")
            if bg_img is None:
                raise IOError(f"Cannot read background image: {bg_img_file}")
            
            bg_img = square_image(bg_img)
            merged_img, label_text = overlay_images(fg_img, bg_img)
            
            output_filename = f"{class_name}-{n}.jpg"
            output_img_path = os.path.join(DEST_IMG_DIR, output_filename)
            cv2.imwrite(output_img_path, merged_img)
            
            label_text_with_class = f"{class_id} {label_text}"
            output_label_path = os.path.join(LABEL_DIR, f"{class_name}-{n}.txt")
            with open(output_label_path, 'w') as f:
                f.write(label_text_with_class)

# ==================== Dataset Splitting ====================

def split_dataset():
    """Split generated dataset into train/valid/test sets"""
    if abs(sum(SPLIT_RATIO) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(SPLIT_RATIO)}")
    
    print("\nSplitting dataset...")
    
    for class_name in OBJ_CLASSES:
        imgs_pattern = os.path.join(DEST_IMG_DIR, f"{class_name}-*.*")
        file_list = glob(imgs_pattern)
        
        if not file_list:
            raise ValueError(f"No images found for class {class_name}")
        
        num_files = len(file_list)
        random.shuffle(file_list)
        
        print(f"\n{class_name}: {num_files} images")
        
        current_idx = 0
        for i, (folder, ratio) in enumerate(zip(DATASET_FOLDERS, SPLIT_RATIO)):
            num_files_in_split = int(num_files * ratio)
            if i == len(SPLIT_RATIO) - 1:
                num_files_in_split = num_files - current_idx
            
            print(f"  {folder}: {num_files_in_split} images")
            
            dest_img_dir = os.path.join(DATASET_PATH, folder, 'images')
            dest_label_dir = os.path.join(DATASET_PATH, folder, 'labels')
            
            for j in range(num_files_in_split):
                if current_idx >= num_files:
                    break
                
                img_file = file_list[current_idx]
                filename = os.path.basename(img_file)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_file = os.path.join(LABEL_DIR, label_filename)
                
                shutil.copy(img_file, dest_img_dir)
                shutil.copy(label_file, dest_label_dir)
                
                current_idx += 1

# ==================== YAML Configuration ====================

def create_yaml_config():
    """Create YAML config file for YOLOv7 training"""
    yaml_content = []
    
    for folder in DATASET_FOLDERS:
        key = 'val' if folder == 'valid' else folder
        path = os.path.join(DATASET_PATH, folder)
        yaml_content.append(f"{key}: {path}")
    
    yaml_content.append('')
    yaml_content.append(f"nc: {len(OBJ_CLASSES)}")
    yaml_content.append('')
    yaml_content.append(f"names: {OBJ_CLASSES}")
    
    yaml_path = os.path.join(CUR_DIR, 'data', 'custom.yaml')
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write('\n'.join(yaml_content))

# ==================== Main ====================

def main():    
    try:
        print("\nStep 1: Creating directory structure")
        create_directory_structure()
        
        print("\nStep 2: Generating synthetic dataset")
        generate_dataset()
        
        print("\nStep 3: Splitting dataset")
        split_dataset()
        
        print("\nStep 4: Creating YAML config")
        create_yaml_config()
        
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()