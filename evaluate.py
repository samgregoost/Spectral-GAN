from GAN_F import GAN_F
from GAN_B import GAN_B
import tensorflow as tf

INPUT_PATH = "input/path"
EPOCHS = 100
LEARNING_RATE = 5e-5
DECAY = 0.9
MOMENTUM = 0.001
EPSILON = 1e-10
SAVE_PATH = "data"

with tf.Graph().as_default():

        gan_f1 = gan(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f2 = gan(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f3 = gan(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f4 = gan(4, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b1 = gan(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b2 = gan(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b3 = gan(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)

        gan_f1.inference()
        gan_f2.inference()
        gan_f3.inference()
        gan_f4.inference()
        gan_b1.inference()
        gan_b2.inference()
        gan_b3.inference()





