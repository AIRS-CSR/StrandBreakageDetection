import os
import cv2
import time
import numpy as np
from process import preprocess_dxdg_seg, postprocess_dxdg_seg
from line_cut import filter_contours, group, combine_contours,  rotate_image_to_align_x_axis
from ONNX import ONNXModel

def draw_parallel_frames(img,rotated_box,cls_box):
    '''
    '''
    A, B, C, D= rotated_box[0], rotated_box[1], rotated_box[2], rotated_box[3]
    x1 = int((A[0]+B[0])/2)
    y1 = int((A[1]+B[1])/2)
    if cls_box == 0:
        cv2.line(img, A, B, (0,255,0), thickness=2)
        cv2.line(img, B, C, (0,255,0), thickness=2)
        cv2.line(img, C, D, (0,255,0), thickness=2)
        cv2.line(img, D, A, (0,255,0), thickness=2)
        cv2.putText(img, str(cls_box), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    elif cls_box == 1:
        cv2.line(img, A, B, (255,0,0), thickness=2)
        cv2.line(img, B, C, (255,0,0), thickness=2)
        cv2.line(img, C, D, (255,0,0), thickness=2)
        cv2.line(img, D, A, (255,0,0), thickness=2)
        cv2.putText(img, str(cls_box), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)


def resize_image_color(image):
    
 
    height, width = image.shape[:2]
    max_size = 1024
    ratio = max_size / max(height, width)
    resized_image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    return resized_image, ratio

def batch_array(array_list, batch_size = 8):
   # Determine the number of batches
    num_batches = (len(array_list) + batch_size - 1) // batch_size
    # Initialize the result array
    packed_arrays = []

    # Pack and pad the arrays
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(array_list))
        batch_arrays = array_list[start_idx: end_idx]

        num_padding = batch_size - len(batch_arrays)
        batch_arrays += [np.zeros([1,3,224,224]) for _ in range(num_padding)]
        packed_arrays.append(np.concatenate(batch_arrays, axis=0).astype(np.float32))

    return packed_arrays

def resize_image(image):
    
    height, width = image.shape[:2]
    max_size = 1024
    ratio = max_size / max(height, width)
    resized_image = cv2.resize(image, (int(width * ratio), int(height * ratio)))
    ret, resized_image = cv2.threshold(resized_image, 5, 255, cv2.THRESH_BINARY)
    return resized_image, ratio

def run_test_dxdg(img0,img_name):
    img, offset_h, offset_w = preprocess_dxdg_seg(img0)
    t3 = time.time()
    pred = model_dxdg_seg.forward(img)
    t4 = time.time()
    print('seg_time', t4 - t3)  
    pred = postprocess_dxdg_seg(pred[0], [img0.shape[0], img0.shape[1]], offset_h, offset_w)
    cv2.imwrite(os.path.join('out',img_name), pred)
    pred, ratio = resize_image(pred)
    res_h, res_w = pred.shape[0], pred.shape[1]
    filter_c, filter_img = filter_contours(pred)
    filter_c.sort(key=lambda x:x['hw_ratio'], reverse=True)
    pair, groups = group(filter_c, res_h, res_w)
    combine_groups = combine_contours(filter_c, groups, ratio, res_h, res_w, filter_img)
    result = {}
    result['img_name'] = img_name
    cls_list = []
    time5 = time.time()
    ori_boxes, rotated_image, rotated_points, rot_crop = rotate_image_to_align_x_axis(combine_groups, img0)
    packed_arrays = batch_array(rot_crop)
    time6 = time.time()
    print('predata:', time6 - time5)
    time7 = time.time()
    for array in packed_arrays:
        crop_cls = model_dxdg_cls.forward(array)
        argmax_indices = np.argmax(crop_cls[1], axis=1)
        cls_list.extend(argmax_indices)
    time8 = time.time()
    print('cls_infer', time8 - time7)
    result['box'] = ori_boxes
    result['cls'] = cls_list
    
        
    return result

if __name__ == '__main__':
    onnx_model_dxdg_seg = "./models/DXDG_512/seg_weights/model.onnx" 
    onnx_model_dxdg_cls = "./models/DXDG_512/cls_weights/model_8.onnx"
    model_dxdg_seg = ONNXModel(onnx_path=onnx_model_dxdg_seg)   
    model_dxdg_cls = ONNXModel(onnx_path=onnx_model_dxdg_cls)
    img_path = "./test_image/" 
    out_path = "./plot_result"
    img_list = os.listdir(img_path)
    t= []
    vis_message = []
    for i, img in enumerate(img_list):
        print('num:{} {}'.format(i, img))
        img0 = cv2.imread(os.path.join(img_path, img))
        img0, ratio = resize_image_color(img0)
        t1 = time.time()
        result = run_test_dxdg(img0, img)
        t2 = time.time()
        infer_time = t2 - t1
        t.append(infer_time)
        vis_message.append(result)
    mean_time = np.mean(t[5:])
    print('Mean inference time per image is:', mean_time)

    for img_ms in vis_message:
        img = cv2.imread(os.path.join(img_path, img_ms['img_name']))
        img, ratio = resize_image_color(img)
        boxes = img_ms['box']
        cls_box = img_ms['cls']
        for i,box in enumerate(boxes):
            draw_parallel_frames(img, box, cls_box[i])
        cv2.imwrite(os.path.join(out_path, img_ms['img_name']), img)