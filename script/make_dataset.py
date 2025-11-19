import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import random, collections, os

BASE = './archive'
FILE_TRAIN = './Train.csv'
TARGET_SIZE = (64, 64)
OUTPUT = './dataset'

def loading(full_path):
    try:
        img = cv2.imread(full_path)
        if img is None:
            raise FileNotFoundError(f'File non trovato: {full_path}')
        return img
    except Exception as e:
        print(f'Errore nel caricamento: {e}')
        return None
    
def rectangle(image, roi, class_id):
    x1, y1, x2, y2 = roi

    display_image = image.copy()
    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Classe: {class_id} - Immagine con ROI')
    plt.axis('off')
    plt.show()

def preprocess(image, roi, target_size):
    x1, y1, x2, y2 = roi

    cropped = image[y1:y2, x1:x2]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    conv = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = conv.astype(np.float32) / 255.0
    return normalized

def distribution(Y):
    count = collections.Counter(Y)
    items = sorted(count.items())
    print('\nDistribuzione')
    for key, value in items:
        print(f'Classe {key:02d}: {value} immagini')
        #print(f'{value}')

def saving(X, Y, output):
    if not os.path.exists(output):
        os.makedirs(output)
    
    np.save(os.path.join(output, 'X.npy'), X)
    np.save(os.path.join(output, 'Y.npy'), Y)
    print(f'Dataset salvato: {output}')

def main():
    try:
        df = pd.read_csv(FILE_TRAIN)
        df.columns = [col.strip() for col in df.columns]
    except FileNotFoundError:
        print(f'File non trovato: {FILE_TRAIN}')
        exit()
    
    X, Y = [], []
    shown = set()

    for i, row in df.iterrows():
        relative_path = row['Path'].replace('Train/', '')
        full_path = os.path.join(BASE, relative_path)
        class_id = row['ClassId']

        roi = (row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
        image = loading(full_path)
        if image is None:
            continue
        
        if class_id not in shown:
            rectangle(image, roi, class_id)
            shown.add(class_id)
        
        process_image = preprocess(image, roi, TARGET_SIZE)
        X.append(process_image)
        Y.append(class_id)
    
    X = np.array(X)
    Y = np.array(Y)

    print(f'\n---- Output finale ----')
    print(f'Image shape: {X.shape}, Label shape: {Y.shape}')
    distribution(Y)
    saving(X, Y, OUTPUT)

if __name__ == '__main__':
    main()