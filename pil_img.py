from PIL import Image,ImageEnhance,ImageChops
import random
import numpy as np

def data(img,img_label):
    def weight_choice(list, weight):
        new_list = []
        for i, val in enumerate(list):
            new_list.extend(val * weight[i])
        return random.choice(new_list)

    switchs = []
    for i in range(4):
        switchs.append(weight_choice(['0', '1'], [8, 2]))

    # print(switchs)
    if switchs[0] == '1':
        a = random.randint(-5, 5)
        img = img.rotate(a)
        img_label = img_label.rotate(a)

    if switchs[1] == '1':
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_label = img_label.transpose(Image.FLIP_LEFT_RIGHT)

    if switchs[2] == '1':
        width, height = img.size
        img = img.crop((int(width / 2) - int(width / 3), int(height / 2) - int(height / 3),
                        int(width / 2) + int(width / 2), int(height / 2) + int(height / 2)))

        img_label = img_label.crop((int(width / 2) - int(width / 3), int(height / 2) - int(height / 3),
                        int(width / 2) + int(width / 2), int(height / 2) + int(height / 2)))

    # if switchs[3] == '1':
    #     xoff = random.randint(-30,30)
    #     yoff = random.randint(-30,30)
    #     width, height = img.size
    #
    #     img = ImageChops.offset(img, xoff, yoff)
    #     img.paste((0, 0, 0), (0, 0, xoff, height))
    #     img.paste((0, 0, 0), (0, 0, width, yoff))
    #
    #     img_label = ImageChops.offset(img_label, xoff, yoff)
    #     img_label.paste((0, 0, 0), (0, 0, xoff, height))
    #     img_label.paste((0, 0, 0), (0, 0, width, yoff))

    return img,img_label

# 指定逆时针旋转的角度（°）
# a = random.randint(-180, 180)
# img_rotate = img.rotate(a)
# img_rotate.show()

# 左右翻转
# out = img.transpose(Image.FLIP_LEFT_RIGHT)
# out.show()
# # 上下翻转
# out = img.transpose(Image.FLIP_TOP_BOTTOM)

####-----------####
#用在原始图片
####-----------####
# 增强因子为0.0产生黑色图像，为1.0保持原始图像
# brightness_factor = np.random.randint(8, 16) / 10
# brightness_image = ImageEnhance.Brightness(img).enhance(brightness_factor)
# brightness_image.show()

#对比度
# contrast_factor = np.random.randint(8, 16) / 10
# contrast_image = ImageEnhance.Contrast(img).enhance(contrast_factor)

#色彩饱和度
# color_factor = np.random.randint(5, 15) / 10
# color_image = ImageEnhance.Color(img).enhance(color_factor)

#锐度
# sharp_factor = np.random.randint(8, 12) / 10
# sharp_image = ImageEnhance.Sharpness(img).enhance(sharp_factor)
####-----------####

#裁剪
# width, height = img.size
# image_crop = img.crop((int(width/2)-int(width/3),int(height/2)-int(height/3),int(width/2)+int(width/2),int(height/2)+int(height/2)))
# image_crop.show()

#平移
# xoff = random.randint(-180,180)
# yoff = random.randint(-180,180)
# width, height = img.size
# c = ImageChops.offset(img, xoff, yoff)
# c.paste((0, 0, 0), (0, 0, xoff, height))
# c.paste((0, 0, 0), (0, 0, width, yoff))
# c.show()

