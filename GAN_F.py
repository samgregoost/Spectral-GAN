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

class GAN_F:
    def __init__(self, fband, nepochs, in_path, l_rate, decay, momentum, epsilon ):
        self.fband = fband
        self.epochs = nepochs
        self.in_path = in_path

        self.l_rate = l_rate
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.X = tf.placeholder(tf.float32, [None, 50 * 50 * 2 * 1])
        self.Z = tf.placeholder(tf.float32, [None, 200 + 50 * 50 * 2 * 1])
        # Z = tf.placeholder(tf.float32,[None,200])
        # Z = tf.placeholder(tf.float32,[None,200+ 50*50*2*8])
        self.Condition1 = tf.placeholder(tf.float32, [None, 50 * 50 * 2 * 1])
        self.Condition2 = tf.placeholder(tf.float32, [None, 50 * 50 * 2 * 1])
        # pregen = tf.placeholder(tf.float32,[None,15*15*2])

        # tf.concat([t3, t4], 0)
        self.x_ = tf.concat([self.Condition1, self.generator(Z)], 1)

        self.g_out = self.generator(Z)

        self.d, _ = discriminator(X)
        self.d_, _ = discriminator(x_, reuse=True)

        self.g_loss = tf.reduce_mean(d_)
        self.d_loss = tf.reduce_mean(d) - tf.reduce_mean(d_)

        self.reg = tc.layers.apply_regularization(
            tc.layers.l1_regularizer(2.5e-1),
            weights_list=[var for var in tf.global_variables() if 'weights' in var.name]
        )

        self.g_loss_reg = g_loss  # + reg
        self.d_loss_reg = d_loss  # + reg

        self.gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
        self.disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.disc_step = tf.train.RMSPropOptimizer(learning_rate=self.l_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon) \
                .minimize(self.d_loss_reg, var_list=self.disc_vars)
            self.gen_step = tf.train.RMSPropOptimizer(learning_rate=self.l_rate, decay=self.decay, momentum=self.momentum, epsilon=self.epsilon) \
                .minimize(self.g_loss_reg, var_list=self.gen_vars)

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.disc_vars]
        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.tf.global_variables_initializer().run(session=sess)

    def get_y(self, x):
        return 10 + x * x

    def generator_(self, Z, hsize=[200, 1000, 1000], reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
            h1 = tf.layers.dense(Z, hsize[0], activation=tf.nn.tanh)
            h2 = tf.layers.dense(h1, hsize[1], activation=tf.nn.tanh)
            h3 = tf.layers.dense(h2, hsize[2], activation=tf.nn.tanh)
            out = tf.layers.dense(h3, 50 * 50 * 2)

        return out


    def generator(self, Z, reuse=False):
        with tf.variable_scope("GAN/Generator", reuse=reuse):
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


    #   return out


    def discriminator(self, X, hsize=[1000, 200, 16], reuse=False):
        with tf.variable_scope("GAN/Discriminator", reuse=reuse):
            h1 = tf.layers.dense(X, hsize[0], activation=tf.nn.leaky_relu)
            h2 = tf.layers.dense(h1, hsize[2], activation=tf.nn.leaky_relu)
            #  h3 = tf.layers.dense(h2,hsize[2],activation=tf.nn.leaky_relu)
            h4 = tf.layers.dense(h2, 2, activation=tf.nn.leaky_relu)
            out = tf.layers.dense(h4, 1)

        return out, h4


    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])


    def dataset_files(self, root):
        """Returns a list of all image files in the given directory"""
        return list(itertools.chain.from_iterable(
            glob(os.path.join(root, "*.{}".format(ext))) for ext in ["npy"]))
    # Z = tf.placeholder(tf.float32,[None,200])
    # Z = tf.placeholder(tf.float32,[None,200+ 50*50*2*8]
    # G_sample = generator(Z)
    # r_logits, r_rep = discriminator(X)
    # f_logits, g_rep = discriminator(G_sample,reuse=True)

    # disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

    # disc_loss =  tf.losses.mean_squared_error(labels = tf.ones_like(r_logits),predictions = r_logits) + tf.losses.mean_squared_error(labels =tf.zeros_like(f_logits), predictions = f_logits)

    # gen_loss = tf.losses.mean_squared_error(labels =tf.ones_like(f_logits), predictions = f_logits)

    # tf.losses.mean_squared_error(
    #    labels,
    #    predictions)


    # gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    # disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    # gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) # G Train step
    # disc_step = tf.train.RMSPropOptimizer(learning_rate=0.00001).minimize(disc_loss,var_list = disc_vars) # D Train step




    # data = dataset_files(config.dataset)
    # np.random.shuffle(data)
    # assert(len(data) > 0)
    # batch_idxs = min(len(data), np.inf) // self.batch_size

    def get_image(self,image_path):
        # print(np.load(image_path).shape)
        return np.load(image_path)


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


    # pcd = PointCloud()


    def inference(self):
        self.saver.restore(sess, ".Models/modelf" + str(self.fband) + ".ckpt")
        if self.band > 1:
            condition1 = np.load(SAVE_PATH + "/harmonics" + str(self.band-1))
        else:
            condition1 = np.zeros((1, 50 * 50 * 2 * 1))

        Z_batch = np.concatenate((condition1, sample_Z(1, 200)), axis=1)

        pred = sess.run([self.g_out],
                          feed_dict={self.Z: Z_batch})

        np.save(pred, SAVE_PATH + "/harmonics" + str(self.band))

    def convert(self, cilm):
        topo = pyshtools.expand.MakeGridDH(cilm, sampling=1)

        theta = np.linspace(0, math.pi, num=300)
        phi = np.linspace(0, 2 * math.pi, num=300)

        theta__, phi__ = np.meshgrid(theta, phi)

        theta_ = theta__.reshape(-1, 1)
        phi_ = phi__.reshape(-1, 1)

        topo_ = topo.reshape(-1, 1)

        final = np.hstack((theta_, phi_, topo_))

        m = sph2cart(final)

        pcd.points = Vector3dVector(m)

        write_point_cloud("g.ply", pcd)
    '''
    saver.restore(sess, "./model.ckpt")
    
    # ata = dataset_files('/media/ram095/329CCC2B9CCBE785/harmonicsch')
    
    # atch_files = data[0]
    # rint(batch_files)
    # X_batch = get_image(batch_files)
    X_batch = np.load('8.npy')
    # X_batch = get_image('/media/ram095/329CCC2B9CCBE785/harmonicsch/200.npy')
    X_batch_ = np.reshape(X_batch, (1, -1))[:, 0:50 * 50 * 2 * 8]
    # print(X_batch_[:,50*50*2*0:50*50*2*1])
    condition1 = np.reshape(X_batch, (1, -1))[:, 0:50 * 50 * 2 * 0]
    condition2 = np.reshape(X_batch, (1, -1))[:, 50 * 50 * 2 * 2:50 * 50 * 2 * 2]
    
    Z_batch = np.concatenate((condition1, sample_Z(1, 200)), axis=1)
    # Z_batch = sample_Z(1, 100)
    
    cilm__ = sess.run([x_], feed_dict={Z: Z_batch, Condition1: condition1, Condition2: condition2})
    
    print(cilm__[0])
    # print(condition)
    
    # cilm_ = np.concatenate((condition, cilm__[0]), axis = 1)
    # cilm = np.reshape(cilm__,(2,150,150))
    # cilm = np.reshape(cilm__,(2,150,150))
    
    # np.save('./9.npy', cilm__[0])
    
    
    topo = pyshtools.expand.MakeGridDH(cilm, sampling=1)
    
    theta = np.linspace(0, math.pi, num=300)
    phi = np.linspace(0, 2*math.pi, num=300)
    
    theta__, phi__ = np.meshgrid(theta, phi)
    
    theta_ = theta__.reshape(-1,1)
    phi_ = phi__.reshape(-1,1)
    
    topo_ = topo.reshape(-1,1)
    
    final = np.hstack((theta_,phi_,topo_))
    
    m = sph2cart(final)
    
    pcd.points = Vector3dVector(m)
    
    write_point_cloud("g4.ply", pcd)
    '''
    def train(self):

        for i in range(self.nepochs):
            data = dataset_files(self.in_path)
            np.random.shuffle(data)
            assert(len(data) > 0)
            batch_idxs = min(len(data), np.inf) // 1
            itr = 0
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*1:(idx+1)*1]
                itr = itr + 1
                X_batch = [get_image(batch_file)
                    for batch_file in batch_files]

                X_batch = np.reshape(X_batch, (1,-1))[:,50*50*2*(self.band-1):50*50*2*self.band]
                #print(X_batch_[:,50*50*2*0:50*50*2*1])
                condition1 = np.zeros((1,50*50*2*1))
                if self.fband > 1:
                    condition1 = np.reshape(X_batch, (1, -1))[:, 50*50*2*(self.band-2):50*50*2*(self.band-1)]
               # condition2 = np.reshape(X_batch, (1,-1))[:,50*50*2*2:50*50*2*2]
              #  condition = np.reshape(X_batch, (1,-1))[:,0:0]
         #   X_batch = sample_data(n=10)

                #Z_batch = sample_Z(1, 200)
                Z_batch = np.concatenate((condition1, sample_Z(1, 200)), axis = 1)
                sess.run(self.d_clip)
                _, dloss = sess.run([self.disc_step, self.d_loss_reg], feed_dict={self.X: X_batch_, self.Z: Z_batch, self.Condition1: condition1, self.Condition2: condition2})
            #    sess.run(d_clip)
                cilm__ = sess.run([self.x_], feed_dict={self.Z: Z_batch, self.Condition1: condition1, self.Condition2: condition2})
               # print(X_batch_)
                #print(cilm__[0][:,50*50*2*0:50*50*2*1])
                _, gloss = sess.run([self.gen_step, self.g_loss_reg], feed_dict={self.Z: Z_batch, self.Condition1: condition1, self.Condition2: condition2})
                if itr%100==0:
                    print("saving model")
                    saver.save(sess, "Models/modelf" + str(self.fband) + ".ckpt")
                print(itr)
                print(dloss)
                print(gloss)

