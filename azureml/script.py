# import some common libraries
import argparse
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import custom scripts
from utils import split_json_train_test, get_fish_dicts, get_labels_from_via

# azureml libraries
from azureml.core import Dataset, Run

# import some common detectron2 utilities
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators
from detectron2.data import build_detection_test_loader

#NUM_CLASSES = 2

def main():
    print("Torch version:", torch.__version__)
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="pytorch-bruv.pt",
                        help='name with which to register your model')
    parser.add_argument('--output_dir', default="local-outputs",
                        type=str, help='output directory')
    parser.add_argument('--n_epochs', type=int,
                        default=10, help='number of epochs')
    parser.add_argument('--dataset_name', default="bruvnet",
                        type=str, help='input directory')
    parser.add_argument('--annotation_file', default="via_region_data.json",
                        type=str, help='file containing vgg annotations (JSON)')
    parser.add_argument('--mountpoint', type=str)
    args = parser.parse_args()

    run = Run.get_context()
    base_path = run.input_datasets[args.dataset_name]
    # In case user inputs a nested output directory
    os.makedirs(name=args.output_dir, exist_ok=True)

    # split complete JSON annotation into train/test
    split_json_train_test(base_path, args.annotation_file, args.output_dir)
    
    # get class labels
    class_labels = get_labels_from_via(base_path, args.annotation_file)
    
    json_head, json_ext = os.path.splitext(args.annotation_file)
    
    # create Detectron2 dataset catalog for train/val
    # use update splitter

    for d in ["train", "val"]:
        DatasetCatalog.register("bruv_" + d, lambda d=d: get_fish_dicts(base_path, 
                                                                    os.path.join(args.output_dir, json_head + '_' + d + json_ext),
                                                                       class_labels))
        MetadataCatalog.get("bruv_" + d).thing_classes=list(class_labels.keys())


    # train to model

    cfg = get_cfg()
    cfg.merge_from_file("mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("bruv_train",)
    cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = args.n_epochs    # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64   # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_labels) # dynamically work out number of classes
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()        

    # score model
    #     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    #     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    #     cfg.DATASETS.TEST = ("bruv_val", )
    #     predictor = DefaultPredictor(cfg)

    # from detectron2.utils.visualizer import ColorMode
    #val_dicts = get_fish_dicts(base_path, os.path.join(args.output_dir, json_head + '_val' + json_ext), class_labels)
    #Evaluation with AP metric
    evaluator = COCOEvaluator("bruv_val", cfg, False, output_dir=os.path.join(args.output_dir, "COCO_evaluator"))
    val_loader = build_detection_test_loader(cfg, "bruv_val")
    result = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    # log metrics
    for item in tqdm(result.get('bbox').items()):
        run.log("BBox-" + item[0], item[1])
    
    for item in tqdm(result.get('segm').items()):
        run.log("Segm-" + item[0], item[1])  
if __name__ == '__main__':
    main()
