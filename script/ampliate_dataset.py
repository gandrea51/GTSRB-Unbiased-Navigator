import numpy as np
import cv2
import collections, os, random

BASE = './dataset'
THRESHOLD = 650
TARGET_SIZE = (64, 64)

def loading(data_dir):
    X = np.load(os.path.join(data_dir, 'X_train.npy'))
    Y = np.load(os.path.join(data_dir, 'y_train.npy'))
    print(f'Dataset originale (Train): {X.shape[0]} immagini')
    return X, Y

def rotate(image):
    angle = random.uniform(-15, 15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated

def zoom(image):
    h, w = image.shape[:2]
    factor = random.uniform(.085, 0.95)
    width, height = int(w*factor), int(h*factor)

    left = random.randint(0, w-width)
    top = random.randint(0, h-height)
    cropped = image[top:top+height, left:left+width]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)

def brightness(image):    
    factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    v_channel = hsv[:, :, 2].astype(np.float32) * factor
    v_channel = np.clip(v_channel, 0, 255)
    hsv[:, :, 2] = v_channel.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def contrast(image):
    factor = random.uniform(0.7, 1.3)
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    
    contrasted = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def saturation(image):
    factor = random.uniform(0.7, 1.3)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    s_channel = hsv[:, :, 1].astype(np.float32) * factor
    s_channel = np.clip(s_channel, 0, 255)
    hsv[:, :, 1] = s_channel.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def shift(image):
    (h, w) = image.shape[:2]
    tx = random.randint(-5, 5)
    ty = random.randint(5, 5)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return shifted

def shear(image):
    (h, w) = image.shape[:2]
    pts1 = np.float32([[5, 5], [w - 5, 5], [5, h - 5]])
    
    pts2 = np.float32([
        [5 + random.randint(-2, 2), 5 + random.randint(-2, 2)], 
        [w - 5 + random.randint(-20, 0), 5 + random.randint(-2, 2)], 
        [5 + random.randint(-2, 2), h - 5 + random.randint(-20, 0)]
    ])
    M = cv2.getAffineTransform(pts1, pts2)
    sheared = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return sheared

def noise(image):
    mean = 0
    stddev = random.uniform(5, 15)
    
    gaussian_noise = np.zeros(image.shape, dtype=np.uint8)
    cv2.randn(gaussian_noise, mean, stddev)
    
    noisy_image = cv2.add(image, gaussian_noise)
    return noisy_image

def operation(image):
    operation = [rotate, zoom, brightness, contrast, saturation, shift, shear, noise]
    
    for o in random.sample(operation, k=random.randint(1, 4)):
        image = o(image)
    return image

def get_distribution(labels):
    return collections.Counter(labels)

def normalizing(image):
    return image.astype(np.float32) / 255.0

def perform(images, number):
    augmented = []
    for _ in range(number):
        img = random.choice(images)
        
        aug = operation(img)
        normalized = normalizing(aug)
        augmented.append(normalized)

    return augmented

def apply_aug(X, Y, treshold, size):
    count = get_distribution(Y)

    X_augmented, y_augmented = [], []
    X_denorm = (X * 255).astype(np.uint8)

    for id in range(43):
        current = count[id]
        if current >= treshold:
            continue

        needed = treshold - current
        print(f'Classe {id:02d}: {current} vere + {needed} da generare')

        images = [X_denorm[i] for i in range(len(Y)) if Y[i] == id]
        
        augmented = perform(images, needed)
        X_augmented.extend(augmented)
        y_augmented.extend([id] * needed)
    
    return np.array(X_augmented), np.array(y_augmented)

def saving(X_init, Y_init, X_aug, Y_aug, output_dir):
    X_new = np.concatenate([X_init, X_aug], axis=0)
    Y_new = np.concatenate([Y_init, Y_aug], axis=0)

    idx = np.arange(len(X_new))
    np.random.shuffle(idx)
    X_new = X_new[idx]
    Y_new = Y_new[idx]

    np.save(os.path.join(output_dir, 'X_augmented.npy'), X_new)
    np.save(os.path.join(output_dir, 'y_augmented.npy'), Y_new)
    print(f'\nDataset augmentato salvato in {output_dir}. Totale: {X_new.shape[0]} immagini')

def main():
    X, Y = loading(BASE)
    X_aug, Y_aug = apply_aug(X, Y, THRESHOLD, TARGET_SIZE)
    saving(X, Y, X_aug, Y_aug, BASE)

if __name__ == '__main__':
    main()