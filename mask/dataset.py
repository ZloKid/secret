import os 
from torchvision.utils import Image
import xmltodict
import numpy as np
import re
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision import tv_tensors  
from torchvision.ops import box_area
from typing import Tuple
import torch

class CustomDataset(Dataset):
    def __init__(self,annotation_path,image_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        image_labels = dict()
        for annotation in os.listdir(self.annotation_path):
            string_to_class = {'with_mask':1,'without_mask':2,'mask_weared_incorrect':3}
            with open(annotation_path+annotation,"r")as f:
                parsed_annotations = xmltodict.parse(f.read())
                filename = parsed_annotations["annotation"]["filename"]
                numbers  = re.findall(r'\d+',filename)
                annotation_index = int(''.join(numbers))
                #print(filename)
                objects = parsed_annotations["annotation"]["object"]
                if isinstance(objects,dict):
                    objects = [objects]
                labels_in_image = []
                for object in objects:
                    class_name= object["name"]
                    #WARNING: need to be round up
                    x,y,w,h = [int(float(val)) for val in list(object["bndbox"].values())]
                    labels_in_image.append((string_to_class.get(class_name),[x,y,w,h]))
                image_labels[annotation_index]=labels_in_image
        self.annotations = image_labels

    def __len__(self):
        return len(os.listdir(self.image_path))

    
        
    def __getitem__(self,idx):
        print(idx)
        coordinates = []
        labels = []
        if idx < 0 or idx > len(os.listdir(self.image_path)):
            raise IndexError("index out of bounds:",idx)
        img_name = "maksssksksss"+str(idx)+".png"  
        img_path = os.path.join(self.image_path,img_name)
        image = read_image(img_path)
        image = image[:3, :, :]
        image = image.float()/255.0
        #print("image: ",image)
        annotations = self.annotations[idx]
        for l,c in annotations:
            coordinates.append(c)
            labels.append(l)
        coordinates = torch.FloatTensor(coordinates)
        #boxes = tv_tensors.BoundingBoxes(data = coordinates,
        #        format = tv_tensors.BoundingBoxFormat.XYWH,canvas_size=(256, 256))
        target = {"boxes":coordinates,
                  "labels":torch.IntTensor(labels).type(torch.int64),
      #TODO: cleanup
                  "image_id":idx,
                  }
        #print("boxes: ",coordinates.shape)
        #print(torch.IntTensor(labels).shape)
        return image,target
          

if __name__ == "__main__":
    ANNOTATION_PATH = '../datasets/archive/new_annotations/'
    IMAGE_PATH = '../datasets/archive/resized_images/'
    test = CustomDataset(ANNOTATION_PATH,IMAGE_PATH)
    print(test.__getitem__(1))
    print(test.__getitem__(0))
