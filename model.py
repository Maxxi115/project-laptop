import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# 'fastercnn_resnet50_fpn', 'fastercnn_mobilenet_v3'

def build_model(backbone: str, num_classes: int):
    if backbone == 'fasterrcnn_resnet50_fpn':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

    else:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    in_feature = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes = num_classes)

    return model
