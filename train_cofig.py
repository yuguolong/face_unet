from nets.unet import mobilenet_unet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image,ImageEnhance,ImageChops,ImageOps
import random
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
from pil_img import data
# np.set_printoptions(threshold='nan')
np.set_printoptions(threshold=np.inf)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-nun_class", type=int,default=7,help="train class")
parser.add_argument("-height",type=int,default=224,help="image height")
parser.add_argument("-weight",type=int,default=224,help="image weight")
parser.add_argument("-log_dir",type=str,default="./logs/",help="save_model_path")
parser.add_argument("-load_weight",type=str,default=True,help="default is load model mobilenet weight")
parser.add_argument("-image_path",type=str,default=".\dataset\jpg",help="load img path")
parser.add_argument("-label_path",type=str,default=".\dataset\png1",help="load label path")
args = parser.parse_args()

NCLASSES = args.nun_class
HEIGHT = args.height
WIDTH = args.weight

def resize_image(image, width, height):
    top, bottom, left, right = (0, 0, 0, 0)
    w, h = image.size
    if h < w:
        dh = w - h
        top = dh // 2
        bottom = dh - top
    else:
        dw = h - w
        left = dw // 2
        right = dw - left
    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)  ##left,top,right,bottom
    image = image.resize((width,height))
    return image

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(args.image_path + '/' + name)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img_label = Image.open(args.label_path + '/' + name)
            #数据增强
            img,img_label = data(img,img_label)

            img = resize_image(img, WIDTH, HEIGHT)
            img = np.array(img)
            img = img / 255.0

            img_label = img_label.convert('L')
            img_label = resize_image(img_label, int(WIDTH / 2), int(HEIGHT / 2))
            img_label = np.array(img_label)

            seg_labels = np.zeros((int(HEIGHT/2), int(WIDTH/2), NCLASSES))
            for c in range(NCLASSES):
                seg_labels[:, :, c] = (img_label[:, :] == c).astype(int)
            seg_labels = np.reshape(seg_labels, (-1, NCLASSES))

            X_train.append(img)
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.categorical_crossentropy(y_true,y_pred,from_logits=True)
    loss = 4 * K.sum(crossloss)/HEIGHT/WIDTH
    return loss

def multi_category_focal_loss1(alpha, gamma=2.0):
    epsilon = 1.e-7
    alpha = tf.constant(alpha, dtype=tf.float32)
    alpha = tf.reshape(alpha, shape=[-1, 1])
    gamma = float(gamma)
    def multi_category_focal_loss1_fixed(y_true, y_pred):
        print(y_pred.shape,y_true.shape)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.matmul(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)
        return loss
    return multi_category_focal_loss1_fixed


if __name__ == "__main__":
    log_dir = args.log_dir
    # 获取model
    model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)

    #加载预训练模型
    if args.load_weight:
        BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/''releases/download/v0.6/')
        model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ('1_0', 224)

        weight_path = BASE_WEIGHT_PATH + model_name
        weights_path = keras.utils.get_file(model_name, weight_path)
        print(weight_path)
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    else:
        model.load_weights("logs/last.h5")
    # model.summary()

    # 打开数据集的txt
    with open(r".\dataset\train.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 保存的方式，1世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss',
                                    save_weights_only=True,
                                    save_best_only=False,
                                    period=1
                                )
    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss',
                            factor=0.5,
                            patience=3,
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1
                        )

    #粗训练
    trainable_layer = 84
    for i in range(trainable_layer):
        model.layers[i].trainable = False
    # 交叉熵
    model.compile(loss=[multi_category_focal_loss1(alpha=[0.5, 1, 1.6, 1.5, 1.2, 1.4, 1.5], gamma=2)],
                  metrics=["accuracy"],optimizer=Adam(lr=1e-4))
    #model.compile(loss = loss,optimizer = Adam(lr=1e-4),metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=2,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])
    model.save_weights(log_dir+'model.h5')

    #精细训练
    trainable_layer = 102
    for i in range(trainable_layer):
        model.layers[i].trainable = True

    # 交叉熵
    model.compile(
        loss=[multi_category_focal_loss1(alpha=[0.5, 1, 1.6, 1.5, 1.2, 1.4, 1.5], gamma=2)],
        metrics=["accuracy"], optimizer=Adam(lr=1e-4))
    # model.compile(loss = loss,optimizer = Adam(lr=1e-5),metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=5,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])
    model.save_weights(log_dir+'last.h5')
