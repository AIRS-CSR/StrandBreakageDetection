import time
import os
import csv
import cv2
import numpy as np
from PIL import Image

from network import Network
import matplotlib.pyplot as plt
def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  
def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 
def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 
def show_results(miou_out_path, hist, Recall, Precision, name_classes, tick_font_size = 12):
    draw_plot_func(Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()
if __name__ == "__main__":
    name_classes    = ["background",'object']
    
    path = 'VOCdevkit_DXDG_2.0_/VOC2007/JPEGImages'
    test_file_path = 'VOCdevkit_DXDG_2.0_/VOC2007/ImageSets/Segmentation/test.txt'
    metrics_out_path = "eval_dxdg/segdec"

    count = False
    net = Network()

    
    undefect_file_list = [os.path.join(path,d.replace('\n','')+'.jpg') for d in open(test_file_path,'r').readlines()]

    defect_sample = 0
    undefect_sample = 0
    preds = []
    labels = []
    time_list = []
    
    for idx,path in enumerate(undefect_file_list):
        image       = Image.open(path)
        r_image,cls_out,time = net.detect_image(image, count=count, name_classes=name_classes)
        time_list.append(time)
        cls_out = cls_out[0]
        labels.append(int(path.split('/')[-1].split('.')[0].split('_')[-1]))
        if cls_out[1]>cls_out[0]:
            preds.append(1)
        else:
            preds.append(0)


        if idx % 100 == 0:
            print("[%d/%d]"%(idx, len(undefect_file_list)))

    hist        = fast_hist(np.array(labels), np.array(preds), len(['defect','normal']))
    Recall      = per_class_Recall(hist)
    Precision   = per_class_Precision(hist)

    show_results(metrics_out_path, hist, Recall, Precision, ['defect','normal'])
    print('time:'+str(1/(sum(time_list)/len(time_list))))
    print('fps:'+str((sum(time_list)/len(time_list))))
