import numpy as np
import tensorflow as tf
import itertools
from glob import glob
import os
import math
from six.moves import xrange
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import pyshtools
from open3d import *
import utils as utils
import point_net
import random

def generator(scope, Z, reuse=False):
    with tf.variable_scope(scope + "GAN/Generator", reuse=reuse):
        fc = Z
        fc = tcl.fully_connected(
            fc, 512,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            weights_regularizer=tc.layers.l2_regularizer(2.5e-2),
            activation_fn=tf.nn.leaky_relu
        )
        fc = leaky_relu(fc)
        fc = tcl.fully_connected(
            fc, 512,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            weights_regularizer=tc.layers.l2_regularizer(2.5e-2),
            activation_fn=tf.nn.leaky_relu
        )
        fc = leaky_relu(fc)
        fc = tcl.fully_connected(
            fc, 512,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            weights_regularizer=tc.layers.l2_regularizer(2.5e-2),
            activation_fn=tf.nn.leaky_relu
        )
        fc = leaky_relu(fc)
        fc = tc.layers.fully_connected(
            fc, 50 * 50 * 2,
            weights_initializer=tf.random_normal_initializer(stddev=0.02),
            weights_regularizer=tc.layers.l2_regularizer(2.5e-2),
            activation_fn=tf.tanh
        )
        return fc

def sample_Z(self, m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def sph2cart(self,coords):
        beta = coords[..., 0]
        alpha = coords[..., 1]
        r = coords[..., 2]

        out = np.empty(beta.shape + (3,))

        ct = np.cos(beta)
        cp = np.cos(alpha)
        st = np.sin(beta)
        sp = np.sin(alpha)
        out[..., 0] = r * st * cp  # x
        out[..., 1] = r * st * sp  # y
        out[..., 2] = r * ct  # z
        return out


Z = tf.placeholder(tf.float32, [None, 200])
pred1 = generator("gf1", Z)

Z2 = np.concatenate((pred1, Z), axis=1)
pred2 = generator("gf2", Z2)

Z3 = np.concatenate((pred2, Z), axis=1)
pred3 = generator("gf3", Z3)

Z4 = np.concatenate((pred3, Z), axis=1)
pred4 = generator("gf4", Z4)

Z5 = np.concatenate((pred4, Z), axis=1)
pred5 = generator("gb3", Z5)

Z6 = np.concatenate((pred5, Z), axis=1)
pred6 = generator("gb2", Z6)

Z7 = np.concatenate((pred6, Z), axis=1)
pred7 = generator("gb1", Z7)


saver = tf.train.Saver()
sess = tf.Session()

var_list1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gf1GAN/Generator")
saver.restore(sess, "./modelf1.ckpt", var_list = var_list1)

var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gf2GAN/Generator")
saver.restore(sess, "./modelf2.ckpt", var_list = var_list2)

var_list3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gf3GAN/Generator")
saver.restore(sess, "./modelf3.ckpt", var_list = var_list3)

var_list4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gf4GAN/Generator")
saver.restore(sess, "./modelf4.ckpt", var_list = var_list4)

var_list5 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gb3GAN/Generator")
saver.restore(sess, "./modelb3.ckpt", var_list = var_list5)

var_list6 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gb2GAN/Generator")
saver.restore(sess, "./modelb2.ckpt", var_list = var_list6)

var_list7 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="gb1GAN/Generator")
saver.restore(sess, "./modelb1.ckpt", var_list = var_list7)

prediction = tf.concat([pred1, pred2, pred3, pred4, pred5, pred6, pred7], axis = 0)

topo = pyshtools.expand.MakeGridDH(prediction, sampling=1)

theta = np.linspace(0, math.pi, num=300)
phi = np.linspace(0, 2 * math.pi, num=300)

theta__, phi__ = np.meshgrid(theta, phi)

theta_ = theta__.reshape(-1, 1)
phi_ = phi__.reshape(-1, 1)

topo_ = topo.reshape(-1, 1)

final = np.hstack((theta_, phi_, topo_))

m = sph2cart(final)

feature_pred = point_net.point_branch(m)


gt_ph = tf.placeholder(tf.float32, [None, 1000, 3])

feature_gt = point_net.point_branch(gt_ph)

loss = tf.losses.mean_squared_error(
    feature_gt, feature_pred
)

opt = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(loss)

def train():
    for i in range(1000):
        Z1 = sample_Z(1, 200)
        gt_clouds = np.load("gt_clouds.py") #give the path of the ground  truth point clouds
        gt_ind = random.randint(0, gt_clouds.shape[0])
        gt_cloud = gt_clouds[gt_ind]


        sess.run([opt], feed_dict={Z: Z1, gt_ph: gt_cloud})

    saver.save(sess, "Model/final_model.ckpt")

def inference():
    saver.restore(sess, "Model/final_model.ckpt")
    Z1 = sample_Z(1, 200)
    pred = sess.run([prediction], feed_dict={Z: Z1})
    np.save("final_hm.py", pred)

