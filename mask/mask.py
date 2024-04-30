import os
import dataset
from torch.utils.data import DataLoader,Subset
from torch import randperm,device,save,load
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as f_weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch,evaluate
from torch.optim import SGD 
from torch.optim.lr_scheduler import StepLR
from utils import collate_fn

PATH_TO_DATASET_IMAGES = '../datasets/archive/resized_images/'
PATH_TO_DATASET_ANNOTATIONS = '../datasets/archive/new_annotations/'
PATH_TO_SAVED_MODEL = '../model_v_0_1.pt'

def get_model_instance_segmentation(num_classes):
    # get number of input features for the classifier
    model  = fasterrcnn_resnet50_fpn_v2(weights = f_weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == "__main__":
    custom_dataset = dataset.CustomDataset(PATH_TO_DATASET_ANNOTATIONS,PATH_TO_DATASET_IMAGES)
    indices = randperm(len(custom_dataset)).tolist()
    train_subset = Subset(custom_dataset,indices[:-170])
    test_subset = Subset(custom_dataset,indices[-170:])
    dataloader_train = DataLoader(train_subset,batch_size = 4,shuffle = True,collate_fn=collate_fn)
    dataloader_test = DataLoader(train_subset,batch_size = 4,shuffle = True,collate_fn=collate_fn)
    print(dataloader_test) 
    num_of_epoch = 15  
    model = get_model_instance_segmentation(4)
    #change to cuda if available
    device_cpu = device('cpu')
    model.to(device_cpu)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.015, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(num_of_epoch):

        train_one_epoch(model, optimizer, dataloader_train, device_cpu, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model,dataloader_test,device_cpu) 
        print(model.fc.weight)
        save(model,str(epoch)+"_"+PATH_TO_SAVED_MODEL)
        if(os.path.exists(str(epoch-1)+"_"+PATH_TO_SAVED_MODEL)):
            os.remove(str(epoch-1)+"_"+PATH_TO_SAVED_MODEL) 

        


