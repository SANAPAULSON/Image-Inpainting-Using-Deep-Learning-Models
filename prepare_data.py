import os
import numpy as np
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "celeba_hq_256"
output_dir = "prepared_data"
image_size = (128, 128)

os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]

train_files, test_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

def preprocess_image(image_path, save_path):

    img = Image.open(image_path)
    img = img.resize(image_size)

    img = np.array(img).astype(np.float32) / 127.5 - 1.0
    img = Image.fromarray(((img + 1.0) * 127.5).astype(np.uint8))

    img.save(save_path)

def process_and_save(image_files, split):
    for img_file in image_files:
        img_path = os.path.join(dataset_path, img_file)
        save_path = os.path.join(output_dir, split, img_file)
        preprocess_image(img_path, save_path)

train_files_limited = train_files[:5000]
val_files_limited = val_files[:100]
test_files_limited = test_files[:100]

print("Processing train images...")
process_and_save(train_files_limited, 'train')
print("Processing val images...")
process_and_save(val_files_limited, 'val')
print("Processing test images...")
process_and_save(test_files_limited, 'test')

print("Dataset preparation complete!")