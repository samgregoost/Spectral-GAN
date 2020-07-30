from GAN_F import GAN_F
from GAN_B import GAN_B
import tensorflow as tf
import spatial_train
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("mode", "train", "mode to operate")

INPUT_PATH = "input/path"
EPOCHS = 100
LEARNING_RATE = 5e-5
DECAY = 0.9
MOMENTUM = 0.001
EPSILON = 1e-10
SAVE_PATH = "data"




def train():
    with tf.Graph().as_default():

        gan_f1 = gan(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gf1")
        gan_f2 = gan(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gf2")
        gan_f3 = gan(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gf3")
        gan_f4 = gan(4, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gf4")
        gan_b1 = gan(3, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gb3")
        gan_b2 = gan(2, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gb2")
        gan_b3 = gan(1, EPOCHS, INPUT_PATH, LEARNING_RATE, DECAY, MOMENTUM, EPSILON, SAVE_PATH, "gb1")

        gan_f1.train()
        gan_f2.train()
        gan_f3.train()
        gan_f4.train()
        gan_b1.train()
        gan_b2.train()
        gan_b3.train()

        spatial_train.train()



def inference():
    spatial_train.inference()


def main(argv=None):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "evaluate":
        inference()

if __name__ == "__main__":
    tf.app.run()
