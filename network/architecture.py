from tensorflow.keras import models, layers

def Gazebo(input_shape, classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(5, 5), padding='same', strides=1, activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(64, kernel_size=(5, 5), padding='same', strides=1, activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Flatten(),
        layers.Dense(120, activation='gelu'),
        layers.Dense(84, activation='gelu'),
        layers.Dense(classes, activation='softmax')
    ])
    return model

def Plion(input_shape, classes):
    model = models.Sequential([
        layers.Input(input_shape),

        layers.Conv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(64, kernel_size=(3, 3), padding='same', strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(128, kernel_size=(3, 3), padding='same', strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),
        
        layers.Flatten(),
        layers.Dense(256, activation='gelu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.25),
        layers.Dense(classes, activation='softmax'),
    ])
    return model

def Creatures(input_shape, classes):
    model = models.Sequential([
        layers.Input(input_shape),

        layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.SeparableConv2D(64, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        layers.Conv2D(256, kernel_size=(3, 3), padding='same', strides=1, activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        layers.Flatten(),
        layers.Dense(256, activation='gelu'),
        layers.Dropout(0.35),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.3),
        layers.Dense(classes, activation='softmax'),
    ])
    return model

def Berlin(input_shape, classes):
    model = models.Sequential([
        layers.Input(input_shape),

        layers.SeparableConv2D(16, kernel_size=(1, 1), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.SeparableConv2D(16, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.SeparableConv2D(32, kernel_size=(1, 1), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.SeparableConv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.SeparableConv2D(64, kernel_size=(1, 1), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.SeparableConv2D(64, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), strides=2),

        layers.SeparableConv2D(128, kernel_size=(1, 1), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.SeparableConv2D(128, kernel_size=(3, 3), padding='same', strides=1, activation='relu', depth_multiplier=1),
        layers.BatchNormalization(),
        layers.AveragePooling2D((2, 2), strides=2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='gelu'),
        layers.Dropout(0.35),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.3),
        layers.Dense(classes, activation='softmax'),
    ])
    return model


def backstreet_block(x, filters):
    branch1 = layers.Conv2D(filters, (1, 1), padding='same', strides=1, activation='relu')(x)

    branch3 = layers.Conv2D(filters, (1, 1), padding='same', strides=1, activation='relu')(x)
    branch3 = layers.Conv2D(filters, (3, 3), padding='same', strides=1, activation='relu')(branch3)

    branch5 = layers.Conv2D(filters, (1, 1), padding='same', strides=1, activation='relu')(x)
    branch5 = layers.Conv2D(filters, (5, 5), padding='same', strides=1, activation='relu')(branch5)

    branch7 = layers.Conv2D(filters, (1, 1), padding='same', strides=1, activation='relu')(x)
    branch7 = layers.Conv2D(filters, (7, 7), padding='same', strides=1, activation='relu')(branch7)

    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters, (1, 1), padding='same', strides=1, activation='relu')(branch_pool)

    output = layers.concatenate([branch1, branch3, branch5, branch7, branch_pool], axis=1)
    return output

