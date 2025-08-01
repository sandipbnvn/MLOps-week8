import os
import random
import shutil
import argparse

def poison_dataset(p):
    """
    Poison the dataset by moving images between cat and dog folders.
    
    Args:
        p (float): Poisoning percentage (0-100)
    """
    train_cats_dir = os.path.join('data', 'train', 'cats')
    train_dogs_dir = os.path.join('data', 'train', 'dogs')
    
    # Calculate number of images to move
    total_images = 2000  # 1000 cats + 1000 dogs
    images_to_move = int(total_images * p / 100)
    images_per_class = images_to_move // 2
    
    print(f"Poisoning level: {p}%")
    print(f"Moving {images_to_move} images total ({images_per_class} from each class)")
    
    # Get list of all images
    cat_images = [f for f in os.listdir(train_cats_dir) if f.endswith('.jpg')]
    dog_images = [f for f in os.listdir(train_dogs_dir) if f.endswith('.jpg')]
    
    # Randomly select images to move
    cats_to_move = random.sample(cat_images, images_per_class)
    dogs_to_move = random.sample(dog_images, images_per_class)
    
    # Move cat images to dog folder
    for img in cats_to_move:
        src = os.path.join(train_cats_dir, img)
        dst = os.path.join(train_dogs_dir, img)
        shutil.move(src, dst)
    
    # Move dog images to cat folder
    for img in dogs_to_move:
        src = os.path.join(train_dogs_dir, img)
        dst = os.path.join(train_cats_dir, img)
        shutil.move(src, dst)
    
    print(f"Moved {len(cats_to_move)} cat images to dog folder")
    print(f"Moved {len(dogs_to_move)} dog images to cat folder")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Poison the dataset')
    parser.add_argument('--p', type=float, required=True, help='Poisoning percentage (0-100)')
    args = parser.parse_args()
    
    poison_dataset(args.p) 