from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

def loading(img_path, size=(32, 32)):
    img = image.load_img(img_path, target_size=size)
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def predict_image(model, img_path):
    img = loading(img_path)
    return np.argmax(model.predict(img, verbose=0))

def predict_batch(model, image_dict):
    return {name: predict_image(model, path) for name, path in image_dict.items()}

def show_result(results):
    for name, pred in results.items():
        print(f"{name}: Classe predetta {pred}")

def shows_predict(image_dict, results):
    for name, path in image_dict.items():
        img = image.load_img(path)
        plt.imshow(img)
        plt.title(f"{name}: Classe predetta {results[name]}")
        plt.axis('off')
        plt.show()

def upload_model(model_path):
    if not os.path.join(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    print(f"Modello caricato: {model_path}")
    return load_model(model_path)

def testing(model, test_set, show_image=True):
    for name, subset in test_set.items():
        print(f"\nRisultati per: {name}")
        result = predict_batch(model, subset)
        show_result(result)
        if show_image:
            shows_predict(subset, result)

def main():
    test_set = {
        "Personal": {
            "Cartello 3": "./images/test/personal/sign_3.jpg",
            "Cartello 9": "./images/test/personal/sign_9.jpg",
            "Cartello 17": "./images/test/personal/sign_17.jpg",
            "Cartello 31": "./images/test/personal/sign_31.jpg",            
            "Cartello 39": "./images/test/personal/sign_39.jpg"
        },
        "Internet": {
            "Cartello 13": "./images/test/internet/sign_13.jpg",
            "Cartello 19": "./images/test/internet/sign_19.jpg",
            "Cartello 25 (doppio)": "./images/test/internet/sign_25_double.jpg",
            "Cartello 25": "./images/test/internet/sign_25.jpeg",
            "Cartello 39": "./images/test/internet/sign_39.jpg"
        }
    }

    model = upload_model("./model/TheGiornalisti.h5")
    testing(model, test_set, show_image=True)

if __name__ == "__main__":
    main()