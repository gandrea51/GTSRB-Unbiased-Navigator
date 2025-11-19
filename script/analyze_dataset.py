import numpy as np
import collections, os
from make_dataset import distribution

BASE = './dataset'

def loading(dir):
    X = np.load(os.path.join(dir, 'X_augmented.npy'))
    Y = np.load(os.path.join(dir, 'y_augmented.npy'))
    print(f'Dataset caricato: {X.shape[0]} immagini')
    return X, Y

def main():
    _, Y = loading(BASE)
    distribution(Y)

if __name__ == '__main__':
    main()