from tensorflow.keras.models import load_model, Model
from tensorflow.keras import layers
import tensorflow as tf

classes = 43

base_model = load_model('./model/Creature.h5')
backbone_out = base_model.get_layer('average_pooling2d').output

for layer in base_model.layers:
    layer.trainable = False

x = backbone_out

x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)

ch = layers.Dense(512, activation='relu')(x)
co = layers.Dense(classes + 1, activation='softmax', name='class_output')(ch)

rh = layers.Dense(512, activation='relu')(x)
ro = layers.Dense(4, activation='linear', name='box_output')(rh)

detection_model = Model(
    inputs=base_model.input,
    outputs=[co, ro]
)

detection_model.compile(
    optimizer='adam', 
    loss={'class_output': 'categorical_crossentropy', 'box_output': 'mse'},
    metrics={'class_output': ['accuracy'], 'box_output': ['mae']}
)
detection_model.summary()
