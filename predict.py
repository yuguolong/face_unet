from nets.unet import mobilenet_unet as Deeplabv3
from PIL import Image,ImageOps
import numpy as np
import random
import copy
import os

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image, nw, nh

NCLASSES = 7
HEIGHT = int(416)
WIDTH = int(416)
random.seed(100)
# class_colors = [[0,0,0],[1,1,1]]
class_colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(NCLASSES)]
# print(class_colors)
model = Deeplabv3(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
# model.load_weights("./logs/ep007-loss0.072-val_loss0.052.h5")
model.load_weights("./logs/ep005-loss0.053-val_loss0.328.h5")
imgs = os.listdir("./img")

for jpg in imgs:
    img = Image.open("./img/" + jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img, nw, nh = letterbox_image(img, [HEIGHT, WIDTH])
    img = np.array(img)
    # print(img.shape)
    img = img / 255.0
    img = img.reshape(-1, HEIGHT, WIDTH, 3)
    pr = model.predict(img)[0]
    print(pr.shape)

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2), NCLASSES)).argmax(axis=-1)
    print(pr.shape)

    pr = Image.fromarray(np.uint8(pr))
    pr = pr.resize((WIDTH, HEIGHT))
    pr = pr.crop(((WIDTH - nw) // 2, (HEIGHT - nh) // 2, (WIDTH - nw) // 2 + nw, (HEIGHT - nh) // 2 + nh))
    pr = np.array(pr)
    seg_img = np.zeros((nh, nw, 3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
    # old_img = np.array(old_img)
    # seg_img = np.array(seg_img)
    # seg_img1 = seg_img
    # w,h,c = old_img.shape
    #
    # for i in range(w):
    #     for j in range(h):
    #         seg_img[i][j][0] = seg_img[i][j][0] * old_img[i][j][0]
    #         seg_img[i][j][1] = seg_img[i][j][1] * old_img[i][j][1]
    #         seg_img[i][j][2] = seg_img[i][j][2] * old_img[i][j][2]
    #
    #         if seg_img1[i][j][0] == 0:
    #             seg_img[i][j][0] = 255
    #             seg_img[i][j][1] = 255
    #             seg_img[i][j][2] = 255
    #
    #         else:
    #             pass
    # seg_img = Image.fromarray(np.uint8(seg_img))


    seg_img = Image.blend(old_img, seg_img, 0.5)
    seg_img.save("./img_out/"+jpg)


