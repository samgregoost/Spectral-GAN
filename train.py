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

        gan_f1 = GAN_F(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f2 = GAN_F(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f3 = GAN_F(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_f4 = GAN_F(4, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b1 = GAN_B(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b2 = GAN_B(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)
        gan_b3 = GAN_B(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH)

        gan_f1.train()
        gan_f2.train()
        gan_f3.train()
        gan_f4.train()
        gan_b1.train()
        gan_b2.train()
        gan_b3.train()

		


