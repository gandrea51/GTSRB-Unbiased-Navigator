import tensorflow as tf
from architecture import Berlin, Gazebo, Plion, Creatures
from trainer import compiling
import matplotlib.pyplot as plt
import sys, io
import numpy as np

def saving(model, filename):
    buffer = io.StringIO()
    sys.stdout = buffer
    model.summary()
    sys.stdout = sys.__stdout__

    summary_text = buffer.getvalue()
    
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.axis('off')
    ax.axis('tight')
    ax.text(0.01, 1.0, summary_text, transform=ax.transAxes, fontsize=8, family='monospace', verticalalignment='top')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f'\nOutput model.summary() salvato in {filename}')
    plt.show()

def main():
    TARGET_SIZE = (64, 64, 3)
    CLASSES = 43
    print('Definizione della rete')
    #model = Gazebo(TARGET_SIZE, CLASSES)
    #model = Plion(TARGET_SIZE, CLASSES)
    #model = Creatures(TARGET_SIZE, CLASSES)
    model = Berlin(TARGET_SIZE, CLASSES)

    model = compiling(model)
    
    #saving(model, filename='Gazebo.png')
    #saving(model, filename='Plion.png')
    #saving(model, filename='Creatures.png')
    #saving(model, filename='Berlin.png')

if __name__ == '__main__':
    main()
