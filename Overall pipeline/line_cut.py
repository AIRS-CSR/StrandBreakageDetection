# coding=utf-8
import cv2
import numpy as np
import os
import math
from process import preprocess_cls_pad

def find_intersection_points(p1, p2, width, height):
    xmin = 0
    ymin = 0
    xmax = width
    ymax = height
    count = 0
    result = []

    if p1[0] == p2[0]:
        if xmin <= p1[0] and p1[0] <= xmax:
            result.append((int(p1[0]), int(ymin)))
            result.append((int(p1[0]), int(ymax)))
            count += 2
    else:
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - k * p1[0]

        if ymin <= k * xmin + b and k * xmin + b <= ymax:
            result.append((int(xmin), int(k * xmin + b)))
            count += 1
        if ymin <= k * xmax + b and k * xmax + b <= ymax:
            result.append((int(xmax), int(k * xmax + b)))
            count += 1
        if xmin <= (ymin - b) / k and (ymin - b) / k <= xmax:
            result.append((int((ymin - b) / k
            ), int(ymin)))
            count += 1
        if xmin <= (ymax - b) / k and (ymax - b) / k <= xmax:
            result.append((int((ymax - b) / k), int(ymax)))
            count += 1

    return result



def draw_parallel_frames(img,rotated_box):
    '''
    根据任意四点坐标, box[A=[x1,y2],B=[x2,y2],C=[x3,y3],D=[x4,y4]], 顺时针ABCD, 给输入图像对应点位画四边形框；

    '''
    A, B, C, D= rotated_box[0], rotated_box[1], rotated_box[2], rotated_box[3]

    cv2.line(img, A, B, (0,255,0), thickness=2)
    cv2.line(img, B, C, (0,255,0), thickness=2)
    cv2.line(img, C, D, (0,255,0), thickness=2)
    cv2.line(img, D, A, (0,255,0), thickness=2)
    return None


def order_points(pts):
    """
    sort rectangle points by clockwise 
    返回顺序: 左下、左上、右上、右下
    """
    sort_x = pts[np.argsort(pts[:, 0]), :]

    Left = sort_x[:2, :]
    Right = sort_x[2:, :]
    # Left sort
    Left = Left[np.argsort(Left[:,1])[::-1], :]
    # Right sort
    Right = Right[np.argsort(Right[:,1]), :]

    return np.concatenate((Left, Right), axis=0)

def crop_rotated_box(output,rotated_box):
    """
    根据任意内接矩形，四点（顺时针）box[[x1,y2],[x2,y2],[x3,y3],[x4,y4]]，从输入图像中，截取图像；
    """
    # 保证变换前后四点坐标顺序一致
    # [1, dy+1] -> [1, 1] -> [1+dx, 1] -> [1+dx, 1+dy]
    rotated_box = np.array(rotated_box,dtype=np.int32)
    box = order_points(rotated_box).tolist()
    # box = [rotated_box[0], rotated_box[1],  rotated_box[2], rotated_box[3]]

    if type(box)==type([]):    
        pts1 = np.float32(box)
        if pts1.shape==(4,2):
            dy=int(np.sqrt((box[0][0]-box[1][0])**2+(box[0][1]-box[1][1])**2))
            dx=int(np.sqrt((box[1][0]-box[2][0])**2+(box[1][1]-box[2][1])**2))   
            # 透视前坐标顺序也是: 左下、左上、右上、右下
            pts2 = np.float32([[1, dy+1],
                          [1, 1],
                          [1+dx, 1],
                          [1+dx, 1+dy]]) 
            M = cv2.getPerspectiveTransform(pts1,pts2)
            dst = cv2.warpPerspective(output, M, (output.shape[1],output.shape[0]))
            # print('output.shape: ',output.shape[1],output.shape[0])   
            target = dst[int(pts2[1][1]):int(pts2[0][1]),int(pts2[1][0]):int(pts2[2][0]),:]
            # w:h ?
            # print('dx/dy: ', dx/dy)
            return True,target
        else:
            print("box shape is wrong ,must be list as (4,2)")
            return False,output
    else:
        print("box type is wrong,must be list as [[x1,y2],[x2,y2],[x3,y3],[x4,y4]]")
        return False,output



def distance(bbox0, bbox1):
    x0, y0 = bbox0
    x1, y1 = bbox1
    dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    return dist

def filter_contours(image):
    
  
    img_size = image.shape
    filter_img = np.zeros_like(image)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filter_contours = []
    for item in contours:
        area = cv2.contourArea(item)
        if area>100:
            cv2.drawContours(filter_img, [item], -1, 255, thickness=cv2.FILLED)
            rect = cv2.minAreaRect(item)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box_h = np.sqrt(np.sum(np.square((box[0] - box[1]))))
            box_w = (np.sqrt(np.sum(np.square((box[0] - box[3]))))+1e-8)
            if box_h > box_w:
                max_box_size = box_h
                min_box_size = box_w
            else:
                max_box_size = box_w
                min_box_size = box_h
            hw_ratio = max_box_size/(min_box_size+1e-8)
            if  (hw_ratio>5 and min_box_size>=2):        
                box_center = (box[0]+box[2])/2    
                mess = {'contour':item, 'box':box, 'hw_ratio':hw_ratio, 'box_center':box_center, 'img_size':img_size, 'max_box_size':max_box_size, 'min_box_size':min_box_size}
                filter_contours.append(mess)
        else:
            cv2.drawContours(filter_img, [item], -1, 0, thickness=cv2.FILLED)
    return filter_contours, filter_img

def group(contours, img_h, img_w):
    confusion_matrix = np.zeros((len(contours), len(contours)))
    pair = []
    groups = []
    for i in range(len(contours)):
        for j in range(len(contours)):
            if i < j:
                mes_i = fit_line(contours[i]['contour'], img_h, img_w, 1)
                mes_j = fit_line(contours[j]['contour'], img_h, img_w, 1)
                if calculate_angle_between_lines(mes_i[1][0], mes_j[1][0]) < 10:
                    p1 = np.mean(mes_i[0], 0)
                    p2 = np.mean(mes_j[0], 0)
                    confusion_matrix[i,j] =  distance(p1, p2)
                else: 
                    confusion_matrix[i,j] =  10000
                
                if confusion_matrix[i,j] < (min(contours[i]['min_box_size'], contours[j]['min_box_size']))*2:
                   
                    if len(groups)==0:
                        groups.append([i,j])
                    elif i in np.unique(np.array([i for item in groups for i in item])) or j in np.unique(np.array([i for item in groups for i in item])):
                        for p, item in enumerate(groups):
                            if i in item:
                                groups[p] = list(set(item+[i,j]))
                            elif j in item:
                                groups[p] = list(set(item+[i,j]))
                                
                    else:
                        groups.append([i,j])
                     
                    
                    pair.append([i,j])                    
                confusion_matrix[j,i] = -1
            if i == j:
                confusion_matrix[i,j] = -1
    
    all = range(len(contours))
    groups_np = np.unique(np.array([i for item in groups for i in item]))
    diff_set = set(all) - set(groups_np)
    diff_set = [[i] for i in diff_set]
    groups.extend(diff_set)
    return pair, groups

def combine_contours(filter_contours, groups, ratio, img_h, img_w, filter_img):
    combine_groups = []
    for item in groups:
        dict_item = {}
        for i, id in enumerate(item):
            if i == 0:
              contour = filter_contours[id]['contour']
              contour_all = contour
            else:
              contour = filter_contours[id]['contour']
              contour_all = np.concatenate((contour_all, contour), axis=0)   
        points, kb = fit_line(contour_all, img_h, img_w, ratio)   
        rect = cv2.minAreaRect(contour_all)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        scan_width = get_width(rect, filter_img)
        box_h = np.sqrt(np.sum(np.square((box[0] - box[1]))))
        box_w = (np.sqrt(np.sum(np.square((box[0] - box[3]))))+1e-8)
        if box_h > box_w:
            max_box_size = box_h
            min_box_size = box_w
        else:
            max_box_size = box_w
            min_box_size = box_h
        hw_ratio = max_box_size/min_box_size
        img_size = filter_contours[id]['img_size']
        if  hw_ratio > 10 or (max_box_size/min(img_size))>0.5:
            box_center = (box[0]+box[2])/2
            side_1 = (box[0] + box[3])/2 if box_h >= box_w else (box[0] + box[1])/2
            side_2 = (box[1] + box[2])/2 if box_h >= box_w else (box[3] + box[2])/2
           
            k = (side_1 - side_2)[1]/((side_1 - side_2)[0]+1e-8)     
            b = (box_center[1] - k*box_center[0])
            
            dict_item['width'] = scan_width/ratio
            dict_item['length'] = max_box_size
            dict_item['points'] = points            
            dict_item['contour'] = contour
            dict_item['box'] = box
            dict_item['hw_ratio'] = hw_ratio
            dict_item['box_center'] = box_center
            dict_item['k-b'] = [k, b]
            combine_groups.append(dict_item)
    return combine_groups


def rotate_image(rect, image):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    center, size, angle = rect
    if size[1] > size[0]:
        angle -= 90
    # 定义旋转角度和缩放比例
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 计算旋转后的图像尺寸
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    rotated_width = int(height * abs_sin + width * abs_cos)
    rotated_height = int(height * abs_cos + width * abs_sin)

    # 调整旋转矩阵中的平移分量，使旋转后的图像居中
    rotation_matrix[0, 2] += rotated_width / 2 - center[0]
    rotation_matrix[1, 2] += rotated_height / 2 - center[1]

    # 执行仿射变换，将图像旋转并补充黑色边界
    rotated_image = cv2.warpAffine(image, rotation_matrix, (rotated_width, rotated_height), borderValue=(0, 0, 0))
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]

    return rotated_image, rotated_box



def get_width(rect, img):
    img, box = rotate_image(rect, img)
    list_count = []
    x_values = [point[0] for point in box]
    y_values = [point[1] for point in box]
    min_x = min(x_values) if min(x_values) > 0 else 0
    max_x = max(x_values) if max(x_values) < img.shape[1] else img.shape[1]
    min_y = min(y_values) if min(y_values) > 0 else 0
    max_y = max(y_values) if max(y_values) < img.shape[0] else img.shape[0]
    rotated_image = img[min_y:max_y, min_x:max_x]
    #cv2.imwrite('1.png', rotated_image)

    for j in range(0,rotated_image.shape[1], 10):
        count = np.sum(rotated_image[:, j]>=5)
        list_count.append(count)
    sorted_list = sorted(list_count)  # 对列表进行排序
    #print('sorted_list', sorted_list)
    sorted_list = [x for x in sorted_list if x != 0] #去掉list前端的0
    if len(sorted_list) <= 3:
        avg_width = np.mean(sorted_list)
    else:
        middle_start = len(sorted_list) // 4  # 计算中间一半元素的起始索引
        middle_end = len(sorted_list) * 3 // 4  # 计算中间一半元素的结束索引
        middle_elements = sorted_list[middle_start:middle_end]  # 获取中间一半元素
        avg_width = sum(middle_elements) / len(middle_elements)  # 计算均值
    return avg_width


def get_rect_vertices(center_x, center_y, width, height, angle):
    angle_rad = math.radians(angle)
    rect_center = (center_x, center_y)
    rect_size = (width, height)
    rect_angle = angle * 180 / math.pi
    # Create a RotatedRect object
    rect = (rect_center, rect_size, rect_angle)

    # Get the four vertices of the RotatedRect object
    pts = cv2.boxPoints(rect)
    pts = pts.astype(int).tolist()

    return pts


def get_overlapping_rectangles(start_point, end_point, overlap_ratio, line_width):
    result_rects = []
    result_xy = []
    theta = math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    line_length = math.sqrt(math.pow(start_point[0] - end_point[0], 2) + math.pow(start_point[1] - end_point[1], 2))
    long_edge_length = 24 * line_width
    short_edge_length = 12 * line_width
    overlap_length = overlap_ratio * long_edge_length
    num_rectangles = math.ceil(line_length / (long_edge_length - overlap_length ))
    offset = long_edge_length - overlap_length

    unit_x = (end_point[0] - start_point[0]) / line_length
    unit_y = (end_point[1] - start_point[1]) / line_length

    for i in range(num_rectangles):
        x_pos = start_point[0] + i * offset * unit_x + long_edge_length / 2 * unit_x
        y_pos = start_point[1] + i * offset * unit_y + long_edge_length / 2 * unit_y

        

        # rect_tl = (int(x_pos - (long_edge_length / 2) * unit_x), int(y_pos + (long_edge_length / 2) * unit_x))
        # rect_tr = (int(x_pos + (long_edge_length / 2) * unit_y), int(y_pos - (long_edge_length / 2) * unit_x))
        # rect_br = (int(x_pos + (long_edge_length / 2) * unit_y + short_edge_length * unit_x), int(y_pos - (long_edge_length / 2) * unit_x + short_edge_length * unit_y))
        # rect_bl = (int(x_pos - (long_edge_length / 2) * unit_y + short_edge_length * unit_x), int(y_pos + (long_edge_length / 2) * unit_x + short_edge_length * unit_y))

        result_rects.append(get_rect_vertices(x_pos,y_pos,long_edge_length,short_edge_length,theta))
        result_xy.append([x_pos, y_pos])

    return result_rects, result_xy, [long_edge_length, short_edge_length]

def calculate_angle_between_lines(k1, k2):
    angle = math.degrees(math.atan(abs((k2 - k1) / (1 + k1 * k2))))
    return angle

def fit_line(point, image_h, image_w, ratio):
    #根据各个联通区域的边缘区域采用最小二乘法拟合出直线端点，斜率和截距   
    #point = np.squeeze(item,1)
    output = cv2.fitLine(point, cv2.DIST_L2, 0, 0.01, 0.01)
    k = output[1] / output[0]
    b = output[3] - k* output[2]
    x0 = -b/(k+1e-8)
    x0 = 0 if x0<= 0 else x0
    x0 = image_w if x0>= image_w else x0
    y0 = (k*x0 + b)
    
    x1 = (image_h-b)/(k+1e-8)
    x1 = 0 if x1<= 0 else x1
    x1 = image_w if x1>= image_w else x1
    y1 = (k*x1 +b)
    points = [tuple([int(x0/ratio), int(y0/ratio)]), tuple([int(x1/ratio), int(y1/ratio)])]
    kb = [float(k), float(b)]
    return points, kb

def cut_line_crop(dict_item):
    '''
    ------------------------------------
    输入dictionary dict_item, 提取数据:
    -中心线两端坐标: 'points' = [(x1,y1),(x2,y2)];
    -中心线斜率和截距: 'k-b' = [k,b];
    -中心线(导线宽度): 'width' = width;
    -中心线(导线长度): 'length' = length;
    ------------------------------------
    输出 rotated_box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]:
    倾斜矩形框的四个顶点坐标, 按顺时针排列在list中: 
    ------------------------------------
    w: 框宽度(Wr)是导线宽度(width)的 w倍; 即 Wr = w*width;
    h: 框高度(Hr)是导线宽度(width)的 h倍; 即 Hr = h*width;
    '''
   
    line_rotated_box = []    # 所有框的坐标信息，用于最后画框
    rotated_box = []    #没有rotate坐标，只表示这是倾斜矩形框的四个顶点坐标(顺时针)

    width = dict_item['width']
    length = dict_item['length']
    if 12*width > length:
        width = width/3
    point1 = dict_item['points'][0]
    point2 = dict_item['points'][1]
    rotated_box, rotated_xy, rotated_wh= get_overlapping_rectangles(point1, point2, 0.2, width)
    return rotated_box, rotated_xy, rotated_wh
        
        
def rotate_image_to_align_x_axis(combine_groups, image):
    
    rot_crop = []
    ori_boxes = []
    rotated_image = []
    rotated_points = []
    for dict_item in combine_groups:
        ori_box, ori_xy, ori_wh = cut_line_crop(dict_item)
        ori_boxes.extend(ori_box)
        ori_xy = np.array(ori_xy, dtype=np.float32)
        # 加载图像
        # 计算直线的斜率

        width = dict_item['width']
        length = dict_item['length']
        if 12*width > length:
            width = width/3
        p1 = dict_item['points'][0]
        p2 = dict_item['points'][1]
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        long_edge_length = int(24 * width)
        short_edge_length = int(12 * width)

        # 计算夹角
        angle = math.degrees(math.atan(k))

        center = (image.shape[1] // 2, image.shape[0] // 2)  # 图像中心点
        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算旋转后的图像大小
        rotated_width = int(abs(image.shape[0] * math.sin(math.radians(angle))) + abs(image.shape[1] * math.cos(math.radians(angle))))
        rotated_height = int(abs(image.shape[0] * math.cos(math.radians(angle))) + abs(image.shape[1] * math.sin(math.radians(angle))))
        rotation_matrix[0, 2] += (rotated_width - image.shape[1]) // 2
        rotation_matrix[1, 2] += (rotated_height - image.shape[0]) // 2

        # 进行图像旋转
        rotated_image = cv2.warpAffine(image, rotation_matrix, (rotated_width, rotated_height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        rotated_points = (np.dot(rotation_matrix[:, :2], ori_xy.T) + rotation_matrix[:, 2:]).transpose().astype(np.int32)
        #防止截图时候出界，补一个黑边
        height = rotated_image.shape[0]
        width = rotated_image.shape[1]
        new_height = int(height + long_edge_length)
        new_width =  int(width + long_edge_length)
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
        new_image[long_edge_length:long_edge_length + height, long_edge_length:long_edge_length + width] = rotated_image
        for box in rotated_points:
            center_x, center_y = box[0], box[1]
            x1 = int(center_x - ori_wh[0] / 2)
            y1 = int(center_y - ori_wh[1] / 2)
            x2 = int(center_x + ori_wh[0] / 2)
            y2 = int(center_y + ori_wh[1] / 2)
            crop = new_image[y1+long_edge_length:y2+long_edge_length, x1+long_edge_length:x2+long_edge_length]
            croped = preprocess_cls_pad(crop, [224, 224])
            rot_crop.append(croped)
    return ori_boxes, rotated_image, rotated_points, rot_crop

if __name__ == '__main__':
    input_folder = "/home/..."
    claster_result_folder = "/home/..."
    crop_result_folder = "/home/..."
    for img in os.listdir(input_folder):
        image_path = os.path.join(input_folder, img)
        image = cv2.imread(image_path, 0)
        color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        filter_c, filter_img = filter_contours(image)
        filter_c.sort(key=lambda x:x['hw_ratio'], reverse=True)
        for item in filter_c:
            
            cv2.drawContours(color_image, [item['box']], 0, (0, 255, 0), 2)
            pass
      
        pair, groups = group(filter_c, image.shape[0], image.shape[1])
        combine_groups = combine_contours(filter_c, groups, 1, image.shape[0], image.shape[1], filter_img)
        i = 0
        for item in combine_groups:
            cv2.drawContours(color_image, [item['box']], 0, (0, 0, 255), 2)
            red = (255, 0, 0)
            cv2.line(color_image, item['points'][0], item['points'][1], red, 4)
            ori_box, rotated_image, rotated_points, rot_crop = rotate_image_to_align_x_axis(item, color_image)
            for point in rotated_points:
                cv2.circle(rotated_image, point, 5, (0,0,255), -1)
            for crop in rot_crop:
                print('crop', crop.shape)
                print('img', img)
                cv2.imwrite(os.path.join(crop_result_folder, img.replace('.', '_'+str(i)+'.')), crop)
                i += 1
        cv2.imwrite(os.path.join(claster_result_folder, img), rotated_image)
