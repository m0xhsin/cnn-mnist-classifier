#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2


def remove_line(image):
    # image shape: (H, W, C)
    means = image.mean(axis=(1, 2))
    mask = means == 255

    if np.any(mask):
        index = np.argmax(mask)
        if index > 0:
            image[index] = image[index - 1]
        elif image.shape[0] > 1:
            image[0] = image[1]

    return image

def load_images(input_folder, h, w, c, needToCorrect):
    images = []
    labels = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(input_folder, filename))
        label = int(filename.split('-')[0])
        if image is not None:
            image = cv2.resize(image, (h, w))
            if needToCorrect == 1:
                image = remove_line(image)
            images.append(image)
            labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    print('shape of image : ', images[0].shape, ' label :', labels[0])
    return images, labels

def check_labels(input_folder):
    labels = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            label = filename.split('-')[0]  
            labels.append(label)
    for num in range(10):
        count = labels.count(str(num))
        print(f'Label {num}: {count} images')
    return labels

def oversample(images, labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    max_count = np.max(counts)
    
    new_images = [images]
    new_labels = [labels]

    for label in unique_labels:
        indices = np.where(labels == label)[0]
        current_count = len(indices)

        if current_count < max_count:
            diff = max_count - current_count
            print(f" - Class {label}: Found {current_count}, adding {diff} duplicates.")
            random_indices = np.random.choice(indices, diff)
            new_images.append(images[random_indices])
            new_labels.append(labels[random_indices])
    final_images = np.concatenate(new_images, axis=0)
    final_labels = np.concatenate(new_labels, axis=0)
    
    return final_images, final_labels

def main():
    if len(sys.argv) != 7:
        print("Usage:")
        print("python3 convert_data.py <image_dir> h w c <output.npz> 0/1")
        sys.exit(1)

    image_dir = sys.argv[1]
    H = int(sys.argv[2])
    W = int(sys.argv[3])
    C = int(sys.argv[4])
    output_npz = sys.argv[5]
    do_clean = int(sys.argv[6])
    labels = check_labels(image_dir)
    X, y = load_images(image_dir, H, W, C,  do_clean)
    X, y = oversample(X, y)
    c = np.count_nonzero(y == 5)
    print(f'Label 5: {c} images')
    np.savez(output_npz, X=X, y=y)
    print(f"Saved dataset to {output_npz}")


if __name__ == "__main__":
    main()
