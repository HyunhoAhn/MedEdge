# MedEdge: Accelerating Object Detection in Medical Images with Edge-Based Distributed Inference

<img src="data/MedEdge.pdf" alt="MedEdge.pdf" />


MedEdge is a framework to accelerate object detection of [Scaled-YOLOv4](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html) by leveraging edge computing in medical image analysis. MedEdge applies a distributed inference technique to Scaled-YOLOv4 in order to exploit the computation resources of both a local computer and the edge server for rapidly detecting COVID-19 abnormalities in chest radiographs.


## About the software

#### Dependencies 
* PyTorch 1.7
* NumPy
* Pillow
* Matplotlib 
* OpenCV
* Triton (edge side)
* TritonClient (grpc)

## Training

```
python3 train.py --batch-size 8 --img 1536 1536 \
    --data data/siim.yaml --cfg yolov4-p7.yaml \
    --weights '' --sync-bn --device 0,1,2,3 \
    --name yolov4_siim
```

The training process of original Scaled-YOLOv4 utilizes Mish-Cuda (self-regularized mon-monotonic activation function) and generates a model suitable for GPUs. We have applied the original Mish function to this train.py in order to generate a Scaled-YOLOv4 model for both CPU and GPU execution.

## Model partitioning

```
python3 model_divider.py -div-point 6 -weights weights/best.pt -img-size 512
```

The command generates a partitioned model with the received slicing point for the local computer and the edge server, respectively. The feasible slicing points in Scaled-YOLOv4 range from one to six.


## Distributed inference
 
For distributed inference, we utilize the PyTorch runtime in the local computer and employ [the NVIDIA Triton inference server](https://developer.nvidia.com/nvidia-triton-inference-server) for the edge, which is open source serving software and optimized for edge deployment. 

### Deployment at the edge server

```
  <model-repository-path>/
    <model-name>/
      1/
        model.onnx
```

A new file including the above contents needs to be created for the model name option of Triton.  

```
docker run --gpus all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
-v <model-repository-path>/<model-name>:/models/model/<model-name> \
nvcr.io/nvidia/tritonserver:21.07-py3 tritonserver \
--model-repository=/models/model/ \
--strict-model-config=false
```

This command deploys the sliced model using Triton at the edge. 

### Deployment at the local computer 

```
python3 distributed_inference.py -local-model saved_models/local_device_model.pt \
-img-size 512 -img-path data/test \
-edge-ip 192.168.0.1 -conf-thres 0.4 -iou-thres 0.5
```

This command performs distributed inference from the local computer to the edge server.
