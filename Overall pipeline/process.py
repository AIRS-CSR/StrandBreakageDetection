import os
import time
import numpy as np
import cv2

def preprocess_cls_pad(img, input_shape, letter_box=True):
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 128, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_dxdg_seg(img,  input_shape=[512, 512], letter_box=False):

    offset_h, offset_w = 0, 0
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 0, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    img = (img - mean) / std
   

    # Step 5: Transpose the image to match PyTorch format (C, H, W)
    img_tensor = img.transpose(2, 0, 1).astype(np.float32)

    # Step 6: Add a batch dimension (if needed)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    return img_tensor, offset_h, offset_w

 
def postprocess_dxdg_seg(image ,input_shape, offset_h, offset_w):
    
    image = np.squeeze(image).astype('uint8')*255
    image = image[offset_h:image.shape[0]-offset_h, offset_w:image.shape[1]-offset_w]
    image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
    
    return image_resized    

   