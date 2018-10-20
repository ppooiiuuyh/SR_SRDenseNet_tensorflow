import os
import time
import tensorflow as tf
from utils import *
from imresize import *
from metrics import *
import matplotlib.pyplot as plt
import pprint
import math
import numpy as np
import sys
import glob
from tqdm import tqdm


class VDSR(object):
# ==========================================================
# class initializer
# ==========================================================
    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.preprocess()
        self.model()
        self.init_model()

# ==========================================================
# preprocessing
# ==========================================================
    def preprocess(self):
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.infer_data = []
        if self.args.type == "eval" : input_setup = input_setup_eval
        elif self.args.type == "demo" : input_setup = input_setup_demo


        if self.args.mode == "train":
            # setup train data
            self.train_data = []
            self.train_label = []

            '''
            # scale augmentation
            scale_temp = self.args.scale
            for s in range(2, 5):
                self.args.scale = s
                train_data_, train_label_ = input_setup(self.args, mode="train")
                self.train_data.extend(train_data_)
                self.train_label.extend(train_label_)
            self.args.scale = scale_temp
            '''
            train_data_, train_label_ = input_setup(self.args, mode="train")
            self.train_data.extend(train_data_)
            self.train_label.extend(train_label_)

            # augmentation (rotation, miror flip)
            self.train_data = augumentation(self.train_data)
            self.train_label = augumentation(self.train_label)

            # setup test data
            self.test_data, self.test_label = input_setup(self.args, mode="test")


        elif self.args.mode == "test":
            self.test_data, self.test_label = input_setup(self.args, mode="test")


        elif self.args.mode == "inference":
            pass


        else:
            assert ("invalid augments. must be in train, test, inference")



# ==========================================================
# build model
# ==========================================================
    def model(self):
        shared_inner_model_template = tf.make_template('shared_model', self.inner_model)

        self.images = tf.placeholder(tf.float32,[None, self.args.patch_size, self.args.patch_size, self.args.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.args.patch_size*self.args.scale, self.args.patch_size*self.args.scale, self.args.c_dim],name='labels')
        self.pred = shared_inner_model_template(self.images)

        self.image_test = tf.placeholder(tf.float32,[1,None,None, self.args.c_dim]  ,name='images_test')
        self.label_test = tf.placeholder(tf.float32, [1, None,None, self.args.c_dim],name='labels_test')
        self.pred_test = shared_inner_model_template(self.image_test)

        if self.args.mode == 'train' or self.args.mode == 'test' : self.other_tensors()



# ===========================================================
# inner model
# ===========================================================
    def inner_model(self,inputs):
    #----------------------------------------------------------------------------------
    # input layer
    #------------------------------------------------------------------------------------------
        global_layer_list = []
        with tf.variable_scope("input") as scope:
            # conv_w = tf.get_variable("conv_%02d_w" % (ii), [3, 3, self.args.c_dim, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
            conv_w = tf.get_variable("conv_input_w", [3, 3, self.args.c_dim, self.args.growth_rate], initializer=tf.contrib.layers.xavier_initializer())
            conv_b = tf.get_variable("conv_input_b", [self.args.growth_rate], initializer=tf.constant_initializer(0))
            layer = tf.nn.bias_add(tf.nn.conv2d(inputs, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
            layer = tf.nn.relu(layer)
            global_layer_list.append(layer)

    # ------------------------------------------------------------------------------------
    # hidden layers
    # -----------------------------------------------------------------------------------
        # =============  dense block ====================================
        for i in range(self.args.num_denseblock):
            with tf.variable_scope("denseblock_%02d" % (i)) as scope:

                #========= dense layer ======================================
                layer_list = [layer]
                for ii in range(self.args.num_denselayer):
                    with tf.variable_scope("denselayer_%02d" % (ii)) as scope:
                        conv_w = tf.get_variable("conv_w", [3, 3, self.args.growth_rate * (ii+1), self.args.growth_rate], initializer=tf.contrib.layers.xavier_initializer())
                        conv_b = tf.get_variable("conv_b", [16], initializer=tf.constant_initializer(0))
                        layer =  tf.nn.conv2d((layer if ii == 0 else tf.concat(layer_list, axis=-1)), conv_w, strides=[1, 1, 1, 1], padding='SAME') + conv_b
                        layer = tf.nn.relu(layer)
                    if ii != self.args.num_denselayer - 1 : layer_list.append(layer)
                global_layer_list.append(tf.concat(layer_list,axis =-1))

    # ------------------------------------------------------------------------------------
    # output layers
    # -----------------------------------------------------------------------------------
        #============== bottleneck layer ================================
        with tf.variable_scope("bottleneck"):
            print(global_layer_list)
            layer = tf.concat(global_layer_list, axis = -1)
            print(layer)

            num_bot_input_features = self.args.num_denseblock * self.args.num_denselayer * self.args.growth_rate + self.args.growth_rate
            conv_w = tf.get_variable("conv_w", [3, 3, num_bot_input_features, 256], initializer=tf.contrib.layers.xavier_initializer())
            conv_b = tf.get_variable("conv_b", [256], initializer=tf.constant_initializer(0))
            layer = tf.nn.bias_add(tf.nn.conv2d(layer, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
            layer = tf.nn.relu(layer)

        #============== upscale layer ===================================
        with tf.variable_scope("upscale"):
            s = self.args.scale
            input_shape = tf.shape(inputs)
            output_shape = [input_shape[0],input_shape[1]*s,input_shape[2]*s,256]
            deconv_w = tf.get_variable("deconv_w", [s,s, 256, 256], initializer=tf.contrib.layers.xavier_initializer())
            deconv_b = tf.get_variable("deconv_b", [256], initializer=tf.constant_initializer(0))
            layer = tf.nn.conv2d_transpose(layer, deconv_w, output_shape=output_shape, strides=[1, s, s, 1], padding='SAME') + deconv_b
            layer = tf.nn.relu(layer)

        #============== reconstruction layer ============================
        with tf.variable_scope("reconstruction"):
            conv_w = tf.get_variable("conv_w", [3, 3, 256, self.args.c_dim], initializer=tf.contrib.layers.xavier_initializer())
            conv_b = tf.get_variable("conv_b", [self.args.c_dim], initializer=tf.constant_initializer(0))
            self.out = tf.nn.bias_add(tf.nn.conv2d(layer, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
        pred = self.out

        return pred
    #----------------------------------------------------------------------------------------



# ============================================================
# other tensors related with training
# ============================================================
    def other_tensors(self):
        self.global_step_tensor = tf.Variable(0, trainable=False, name="global_step")

        #optimizer
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.pred)) #L1 is betther than L2
        self.learning_rate = tf.maximum(tf.train.exponential_decay(self.args.base_lr, self.global_step_tensor,
                                                        len(self.train_data)//self.args.batch_size * self.args.lr_step_size,
                                                        self.args.lr_decay_rate, staircase=True),self.args.lr_min) #stair case showed better result
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.global_step_tensor)
        self.summary_writer = tf.summary.FileWriter("./board", self.sess.graph)


    def init_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=0)
        if self.cpkt_load(self.args.checkpoint_dir, self.args.cpkt_itr):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")






# ==========================================================
# train
# ==========================================================
    def train(self):
        self.test()
        print("Training...")
        start_time = time.time()

        # shuffle
        seed = 10
        np.random.seed(seed) ;  np.random.shuffle(self.train_data)
        np.random.seed(seed) ;  np.random.shuffle(self.train_label)

        for ep in range(self.args.epoch):
            # Run by batch images
            batch_idxs = len(self.train_data) // self.args.batch_size

            for idx in tqdm(range(0, batch_idxs)):
                batch_images = self.train_data[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]
                batch_labels = self.train_label[idx * self.args.batch_size: (idx + 1) * self.args.batch_size]

                feed_dict = {self.images: batch_images, self.labels: batch_labels}
                _,err, lr = self.sess.run([self.train_op,self.loss, self.learning_rate], feed_dict=feed_dict)

            if ep % 1 == 0:
                print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], lr: [%.8f]" \
                      % ((ep+1), self.global_step_tensor.eval(self.sess),time.time() - start_time,  np.mean(err),lr))
                self.test()

            if ep % self.args.save_period == 0:
                self.cpkt_save(self.args.checkpoint_dir, ep+1)




# ==========================================================
# test
# ==========================================================
    def test(self):
        print("Testing...")
        psnrs_bicubic = []
        psnrs_preds = []
        ssims_bicubic = []
        ssims_preds = []

        preds = []
        labels = []
        images = []


        # run SR model with each depths
        for idx in range(0, len(self.test_data)):
            image_test = np.array(self.test_data[idx])
            label_test = np.array(self.test_label[idx])
            preds.append(self.inference(image_test, scale=1)) #scale should be set 1 since images already have been upscalsed
            labels.append(label_test)
            images.append(image_test)


        # cal PSNRs for each images upscaled from different depths
        for i in range(len(self.test_data)):
            labels[i] = np.mean(labels[i],axis=-1)
            images[i] = np.mean(images[i],axis=-1)

            preds[i] = np.mean(preds[i],axis=-1)
            psnrs_preds.append(psnr(labels[i], preds[i], max=1.0, scale=self.args.scale))
            ssims_preds.append(ssim(labels[i], preds[i], max=1.0, scale=self.args.scale))


        # print evalutaion results
        print("===================================================================================")
        print("Model " + "PSNR: " + str(round(np.mean(np.clip(psnrs_preds,0,100)), 3)) + "dB")
        print("Model " + "SSIM: " + str(round(np.mean(np.clip(ssims_preds,0,100)), 3)))
        print("===================================================================================")


# ==========================================================
# inference. return upscaled img with getting input image. input image should be numpy format
# ==========================================================
    def inference(self, input_img, scale):
        infer_image_scaled = imresize(input_img, scalar_scale=scale, output_shape=None, mode="vec")
        size = infer_image_scaled.shape
        infer_image_input = infer_image_scaled.reshape(1, size[0], size[1], size[2])
        sr_img = self.sess.run(self.pred_test, feed_dict={self.image_test: infer_image_input})
        return sr_img[0]


# ==========================================================
# test with plotting. using single custom image.
# ==========================================================
    def test_plot(self, infer_image):
        print("inferring...")

        img_size = infer_image.shape
        print("img shape : ", img_size)
        s = 200
        bx = 100
        by = 100
        image_pos_x = [bx, bx+s]
        image_pos_y = [by, by+s]
        image_pos_x = [0, -1]
        image_pos_y = [0, -1]

        images = []
        times = [0 for _ in range(3)]
        for k in range(5):
            infer_image_ = infer_image
            #HR
            f1 = plt.figure(1)
            infer_image_croped = infer_image_[image_pos_x[0]:image_pos_x[1], image_pos_y[0]:image_pos_y[1], :]
            images.append(infer_image_croped)

            #LR
            infer_image_downscaled = imresize(infer_image_, scalar_scale=1/self.args.scale, output_shape=None, mode="vec")
            infer_image_croped = infer_image_downscaled[image_pos_x[0]//self.args.scale:image_pos_x[1]//self.args.scale, image_pos_y[0]//self.args.scale:image_pos_y[1]//self.args.scale, :]
            images.append(infer_image_croped)

            # bicubic
            start_time = time.time()
            infer_image_scaled = imresize(infer_image_downscaled, scalar_scale=self.args.scale, output_shape=None, mode="vec")
            elapse = time.time() - start_time
            if k != 0: times[0] += elapse

            infer_image_croped = infer_image_scaled[image_pos_x[0]:image_pos_x[1], image_pos_y[0]:image_pos_y[1], :]
            images.append(infer_image_croped)

            # MODEL
            start_time = time.time()
            result = self.inference(infer_image_downscaled, scale= self.args.scale)
            elapse = time.time() - start_time
            if k != 0: times[i+1] += elapse

            infer_image_croped = result[image_pos_x[0]:image_pos_x[1], image_pos_y[0]:image_pos_y[1], :]
            images.append(infer_image_croped)



        #print elapses
        print("=========================================================")
        print("Elapse(", k, "):", np.array(times) / k)


        #plot images
        plt.subplot(4, 1, 1)
        plt.title("Original HR")
        plt.imshow(images[0])

        plt.subplot(4, 1, 2)
        plt.title("Original LR")
        plt.imshow(images[1])

        plt.subplot(4, 1, 3)
        plt.title("Bicubic")
        plt.imshow(images[2])

        plt.subplot(4, 1, 4)
        plt.title("Models")
        plt.imshow(images[3])

        f1 = plt.figure(1)
        plt.imshow(images[0])
        f2 = plt.figure(2)
        plt.imshow(images[1])
        f3 = plt.figure(3)
        plt.imshow(images[2])
        f4 = plt.figure(4)
        plt.imshow(images[3])
        plt.show()








# ==========================================================
# GUI DEMO
# ==========================================================
    def demo(self):
        pass





# ==========================================================
# functions
# ==========================================================
    def cpkt_save(self, checkpoint_dir, step):
        model_name = "checks.model"
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def cpkt_load(self, checkpoint_dir, checkpoint_itr):
        print(" [*] Reading checkpoints...")
        model_dir = "checks"
        checkpoint_dir = os.path.join(checkpoint_dir, self.args.type, model_dir)

        if checkpoint_itr == 0:
            print("train from scratch")
            return True

        elif checkpoint_dir == -1:
            ckpt = tf.train.latest_checkpoint(checkpoint_dir)

        else:
            ckpt = os.path.join(checkpoint_dir, "checks.model-" + str(checkpoint_itr))

        print(ckpt)
        if ckpt:
            self.saver.restore(self.sess, ckpt)
            return True
        else:
            return False

# ==========================================================
# others
# ==========================================================


