import os
import shutil
from math import ceil

def split_dataset(source_folder, target_folder, num_buckets=8):
    # Create target buckets if they don't exist
    for i in range(num_buckets):
        os.makedirs(f"{target_folder}/bucket_{i}", exist_ok=True)

    # List all class directories in the source folder
    classes = [d for d in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, d))]

    for cls in classes:
        cls_path = os.path.join(source_folder, cls)
        images = os.listdir(cls_path)

        # Calculate number of images per bucket for this class
        images_per_bucket = ceil(len(images) / num_buckets)

        for i in range(num_buckets):
            # Create class directory in each bucket
            bucket_cls_path = os.path.join(target_folder, f"bucket_{i}", cls)
            os.makedirs(bucket_cls_path, exist_ok=True)

            # Determine the slice of images for this bucket
            start_idx = i * images_per_bucket
            end_idx = min((i + 1) * images_per_bucket, len(images))

            # Copy images to the bucket
            for img in images[start_idx:end_idx]:
                shutil.copy2(os.path.join(cls_path, img), bucket_cls_path)

if __name__ == "__main__":
    source_folder = "/scratch/lprfenau/datasets/imagenet/val"
    target_folder = "/scratch/a2diaa/datasets/split_imagenet/split_val"
    split_dataset(source_folder, target_folder)
