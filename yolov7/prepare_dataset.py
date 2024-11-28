import os
from pathlib import Path
import yaml
import shutil
import random
from tqdm import tqdm

def organize_dataset(base_path):
    random.seed(42)  # For reproducibility
    
    # Define class mapping
    class_mapping = {
        'Jab-annotated': 0,
        'Cross-annotated': 1,
        'Hook-annotated': 2,
        'Uppercut-annotated': 3
    }
    
    # Create output directories
    output_path = Path('yolo_dataset')
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Initialize counters
    total_train_images = 0
    total_val_images = 0
    class_counts = {'train': {}, 'val': {}}
    
    print("\nStarting dataset organization...")
    
    # Process each class directory
    for class_dir, class_idx in class_mapping.items():
        print(f"\nProcessing {class_dir} (class index: {class_idx})")
        
        # Setup paths
        class_path = Path(base_path) / class_dir
        images_path = class_path / "images" / "train"
        labels_path = class_path / "labels" / "train"
        
        # Verify directories exist
        if not images_path.exists() or not labels_path.exists():
            print(f"Warning: Missing directories for {class_dir}")
            continue
        
        # Get all image files
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        
        if not image_files:
            print(f"No images found for {class_dir}")
            continue
            
        print(f"Found {len(image_files)} images for {class_dir}")
        
        # Split into train/val
        random.shuffle(image_files)
        split_idx = int(len(image_files) * 0.8)
        train_images = image_files[:split_idx]
        val_images = image_files[split_idx:]
        
        # Process training images
        print(f"Copying {len(train_images)} training images and labels...")
        for img_path in tqdm(train_images, desc=f"Training {class_dir}"):
            label_path = labels_path / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Copy image
            dest_img = output_path / 'train' / 'images' / img_path.name
            shutil.copy(str(img_path), str(dest_img))
            
            # Copy and modify label
            dest_label = output_path / 'train' / 'labels' / f"{img_path.stem}.txt"
            with open(label_path, 'r') as f_in, open(dest_label, 'w') as f_out:
                for line in f_in:
                    coords = line.strip().split()[1:]
                    if len(coords) == 4:
                        f_out.write(f"{class_idx} {' '.join(coords)}\n")
            
            total_train_images += 1
            class_counts['train'][class_idx] = class_counts['train'].get(class_idx, 0) + 1
        
        # Process validation images
        print(f"Copying {len(val_images)} validation images and labels...")
        for img_path in tqdm(val_images, desc=f"Validation {class_dir}"):
            label_path = labels_path / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Copy image
            dest_img = output_path / 'val' / 'images' / img_path.name
            shutil.copy(str(img_path), str(dest_img))
            
            # Copy and modify label
            dest_label = output_path / 'val' / 'labels' / f"{img_path.stem}.txt"
            with open(label_path, 'r') as f_in, open(dest_label, 'w') as f_out:
                for line in f_in:
                    coords = line.strip().split()[1:]
                    if len(coords) == 4:
                        f_out.write(f"{class_idx} {' '.join(coords)}\n")
            
            total_val_images += 1
            class_counts['val'][class_idx] = class_counts['val'].get(class_idx, 0) + 1
    
    # Create dataset.yaml
    yaml_content = {
        'train': str(output_path / 'train' / 'images'),
        'val': str(output_path / 'val' / 'images'),
        'nc': 4,
        'names': ['Jab', 'Cross', 'Hook', 'Uppercut']
    }
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    
    # Print final statistics
    print("\nDataset Organization Complete!")
    print("\nFinal Dataset Statistics:")
    print(f"Total training images: {total_train_images}")
    print(f"Total validation images: {total_val_images}")
    
    print("\nTraining set class distribution:")
    for class_idx, count in class_counts['train'].items():
        class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_idx)]
        print(f"  {class_name}: {count} images")
    
    print("\nValidation set class distribution:")
    for class_idx, count in class_counts['val'].items():
        class_name = list(class_mapping.keys())[list(class_mapping.values()).index(class_idx)]
        print(f"  {class_name}: {count} images")
    
    print(f"\nDataset YAML file created at: {yaml_path}")
    print(f"Dataset organized in: {output_path}")
    
    # Verify the organization
    print("\nVerifying dataset organization...")
    for split in ['train', 'val']:
        split_path = output_path / split
        n_images = len(list((split_path / 'images').glob('*.jpg'))) + len(list((split_path / 'images').glob('*.png')))
        n_labels = len(list((split_path / 'labels').glob('*.txt')))
        print(f"{split} set: {n_images} images and {n_labels} label files")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Get the base path from command line or use default
    base_path = sys.argv[1] if len(sys.argv) > 1 else 'boxing_dataset'
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Dataset path {base_path} does not exist!")
        sys.exit(1)
    
    print(f"Processing dataset from: {base_path}")
    organize_dataset(base_path)