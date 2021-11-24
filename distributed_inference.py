import argparse
import os
import pickle
import time
from collections.abc import Iterable
from pathlib import Path
import struct

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
import cv2


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    coco80_to_coco91_class, check_file, check_img_size, compute_loss, non_max_suppression,
    scale_coords, xyxy2xywh, clip_coords, plot_images, xywh2xyxy, box_iou, output_to_target, ap_per_class,plot_one_box)
from utils.torch_utils import select_device, time_synchronized    


import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Offloader for distributed inference")

    parser.add_argument('-local-model',  type=str, default="saved_models/local_device_model.pt", required=False,
                        help='local device model file path')
    parser.add_argument('-img-size',  type=int , default=512, required=False,
                        help='inference size (pixels)') 
    parser.add_argument('-img-path',  type=str, default='./data/test', required=False,
                        help='image path')
    parser.add_argument('-output', type=str, default='data/output', help='output folder')  # output folder  
    parser.add_argument('-edge-ip',  type=str, required=True,
                        help='ip address of the edge server ex)192.168.0.1')
    parser.add_argument('-conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('-iou-thres', type=float, default=0.5, help='IOU threshold for NMS')                        
    parser.add_argument('-view-img', action='store_true', help='display results')
    parser.add_argument('-save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('-classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')

    args = parser.parse_args()

    server = args.edge_ip + ":8001" # we use grpc
    img_sz = args.img_size
    view_img, save_txt = args.view_img, args.save_txt
    out = args.output
    os.makedirs(out,exist_ok=True)  # make new output folder

    save_img = True
    dataset = LoadImages(args.img_path, img_size=img_sz) # dataloader

    device_model = torch.load(args.local_model,  map_location=torch.device('cpu')) #load the local device model 
    device_model.eval()
    with torch.no_grad():
        data = device_model(torch.zeros((1, 3, img_sz, img_sz)))
    size = [data.size()[0], data.size()[1],data.size()[2],data.size()[3]] 
    # Get names and colors
    names =  device_model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    #print(names)

    #set up the grpc setting
    channel_opt = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
    channel = grpc.insecure_channel(server,options = channel_opt)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)# Send request

    request = service_pb2.ModelInferRequest()
    request.model_name = "edge_model" # name of the model loaded on the edge server

    input = service_pb2.ModelInferRequest().InferInputTensor()
    input.name = "images"
    input.datatype = "FP32"
    input.shape.extend(size)
    request.inputs.extend([input])

    output_name = 'output'
    output_names = output_name.split(',')
    out_len = len(output_names)
    if out_len > 1:
        for i in range(out_len):
            output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
            output.name = output_names[i]
            request.outputs.extend([output])
    else:
        output = service_pb2.ModelInferRequest().InferRequestedOutputTensor()
        output.name  = output_name
        request.outputs.extend([output])


    for path, img, im0s, vid_cap in dataset:

        request.ClearField("raw_input_contents")

        img = torch.from_numpy(img).to('cpu')
        img = img.float()  
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():
            data = device_model(img)

        data = data.detach().numpy()
        input_bytes = data.tobytes() #convert to bytes
        request.raw_input_contents.extend([input_bytes])
        response = grpc_stub.ModelInfer(request) #offloading the remaining inference task to the edge 


        pred = np.frombuffer(response.raw_output_contents[0], dtype=np.float32)
        pred = np.reshape(pred, response.outputs[0].shape)
        #print(pred.shape)
        pred = torch.from_numpy(pred)
        #print(pred[0][9][2])
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=args.conf_thres, iou_thres=args.iou_thres, agnostic=False)
        #print(len(pred))

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s' % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)


            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

