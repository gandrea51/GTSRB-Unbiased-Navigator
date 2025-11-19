from sklearn.model_selection import train_test_split
import numpy as np
import collections, os

BASE = './dataset'
TEST_RATIO = 0.15
VAL_RATIO = 0.15

def loading(dir):
    X = np.load(os.path.join(dir, 'X.npy'))
    Y = np.load(os.path.join(dir, 'Y.npy'))
    print(f'Dataset caricato: {X.shape[0]} immagini')
    return X, Y

def divide(X, Y, test_size, val_size, state=42):
    if 1.0 - test_size <= 0:
        raise ValueError('Test size deve essere minore di 1')

    val_relative = val_size / (1.0 - test_size)
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(
        X, Y, 
        test_size=test_size, 
        stratify=Y,
        random_state=state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_temp, y_train_temp, 
        test_size=val_relative, 
        stratify=y_train_temp, 
        random_state=state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def saving(output, X_train, X_val, X_test, y_train, y_val, y_test):
    if not os.path.exists(output):
        os.makedirs(output)
    
    np.save(os.path.join(output, 'X_train.npy'), X_train)
    np.save(os.path.join(output, 'y_train.npy'), y_train)
    np.save(os.path.join(output, 'X_val.npy'), X_val)
    np.save(os.path.join(output, 'y_val.npy'), y_val)
    np.save(os.path.join(output, 'X_test.npy'), X_test)
    np.save(os.path.join(output, 'y_test.npy'), y_test)

def main():
    X, Y = loading(BASE)
    X_train, X_val, X_test, y_train, y_val, y_test = divide(X, Y, TEST_RATIO, VAL_RATIO)
    saving(BASE, X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == '__main__':
    main()