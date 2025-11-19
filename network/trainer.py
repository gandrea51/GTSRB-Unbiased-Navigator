import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def loading(dir):
    X_train = np.load(f'{dir}/X_augmented.npy')
    y_train = np.load(f'{dir}/y_augmented.npy')
    X_val = np.load(f'{dir}/X_val.npy')
    y_val = np.load(f'{dir}/y_val.npy')
    X_test = np.load(f'{dir}/X_test.npy')
    y_test = np.load(f'{dir}/y_test.npy')

    classes = int(np.max(y_train)) + 1 
    print(f"Numero di classi inferito: {classes}")

    y_train = tf.keras.utils.to_categorical(y_train, classes)
    y_val = tf.keras.utils.to_categorical(y_val, classes)
    y_test = tf.keras.utils.to_categorical(y_test, classes)

    return X_train, y_train, X_val, y_val, X_test, y_test, classes

def compiling(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def training(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=256):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
    return history

def evaluating(model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose=0)
    print('\nTest loss:', score[0])
    print('Test accuracy:', score[1])
    return score

def plotting(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Validation acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.show()

def confusioning(model, X_test, y_test, labels=None):    
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    confusion = confusion_matrix(y_true_classes, y_pred_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels if labels is not None else range(confusion.shape[0]),
                yticklabels=labels if labels is not None else range(confusion.shape[0]))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_true_classes, y_pred_classes, target_names=labels))