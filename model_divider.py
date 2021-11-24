import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn


from models.experimental import attempt_load
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class)
from utils.torch_utils import select_device, time_synchronized, scale_img, copy_attr


class Edge_model(nn.Module):
    def __init__(self, edge_layers, save_list):  # model, input channels, number of classes
        super(Edge_model, self).__init__()
        self.model = edge_layers  # load edge model's layers
        #self.model[-1].export = True # set Detect() layer to disable
        self.identity = torch.nn.Identity()
        self.save = save_list 
        #print(self.save)

    def forward(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x)  # single-scale inference, train

    def check_backward_forward(self, from_num):
        if from_num < 0:
            pass
        else:
            from_num = from_num - d_v
        return from_num

    def forward_once(self, x):
        y = []  # outputs
        
        x = self.identity(x)
        y.append(x)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[self.check_backward_forward(m.f)] if isinstance(m.f, int) else [x if j == -1 else y[self.check_backward_forward(j)] for j in m.f]  # from earlier layers
    
            x = m(x)  # run
            y.append(x if (m.i-d_v) in self.save else None)  # save output

        return x



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model divider for Edge offloading")


    parser.add_argument('-weights',  type=str, default="weights/best.pt", required=False,
                        help='weights file path to divide')
    parser.add_argument('-img-size',  type=int, default=512, required=False,
                        help='image size') 
    parser.add_argument('-div-point',  type=int, default=6, required=False,
                        help='dividing point (1~6)')                    

    args = parser.parse_args()

    model = attempt_load(args.weights, map_location=torch.device('cpu'))  # onnx needs fuse 

    names = model.module.names if hasattr(model, 'module') else model.names

    img = torch.zeros((1, 3, args.img_size,args.img_size))  

    y = model(img)  # dry run
    
    d_v = args.div_point -1   # dividing_point

    device_model = torch.nn.Sequential(*list(model.model.children())[:d_v+1]) 
    device_model.names = names
    os.makedirs("./saved_models",exist_ok=True) 
    torch.save(device_model,"./saved_models/local_device_model.pt")

    edge_layers = torch.nn.Sequential(*list(model.model.children())[d_v+1:]) #save edge layers
    
    #print(model.save)
    save_list = []   # save save_list for skip connection
    for i in model.save:
        i = i - d_v 
        save_list.append(i)

    edge_model = Edge_model(edge_layers,save_list) 

    edge_model.eval()
    device_model.eval()

    device_output = device_model(img)
    edge_output = edge_model(device_output)


    # edge model onnx export
    try:
        import onnx

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = "./saved_models/edge_model.onnx"  # filename

        torch.onnx.export(edge_model, device_output, f, verbose=False, opset_version=12, input_names=['images'],
                        output_names=['classes', 'boxes'] if y is None else ['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        #print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
    
