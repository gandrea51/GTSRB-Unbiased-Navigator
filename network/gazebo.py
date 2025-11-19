from trainer import loading, compiling, training, evaluating, plotting, confusioning
from architecture import Gazebo

BASE = './dataset'
TARGET_SIZE = (64, 64, 3)

def main():
    X_train, Y_train, X_val, Y_val, X_test, Y_test, classes = loading(BASE)

    model = Gazebo(TARGET_SIZE, classes)
    model = compiling(model)
    model.summary()

    history = training(model, X_train, Y_train, X_val, Y_val)
    evaluating(model, X_test, Y_test)
    plotting(history)
    confusioning(model, X_test, Y_test)

    model.save("./model/Gazebo.h5")

if __name__ == "__main__":
    main()