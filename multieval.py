import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
from torch.optim.lr_scheduler import OneCycleLR
import timm
from transformers import AutoModelForImageClassification, AutoModel
#!export HF_ENDPOINT=https://hf-mirror.com
from datetime import datetime
import time
from utils.utils import set_seed, load_csv
from utils.config import config_args
from utils.dataset import ButterflyDataset
import itertools
from models.baseline import ButterflyCNN as ViT
from models.unireplknet import load_state_dict

from models.MetaFG.MetaFG import *
from models.MetaFG.MetaFG_meta import *
from models.MetaFormer import load_pretained
from models.MPSA.setup import config
from models.MPSA.models.build import build_models

import models.convnext
import models.convnext_isotropic

from timm.models import create_model

from collections import Counter

def most_common_elements(list_of_lists):
    # 获取列表的长度
    list_length = len(list_of_lists[0])
    
    # 初始化结果列表
    result = []
    
    # 遍历每个位置
    for i in range(list_length):
        # 获取当前位置的所有元素
        elements = [lst[i] for lst in list_of_lists]
        
        # 统计每个元素的出现次数
        counter = Counter(elements)
        
        # 获取出现次数最多的元素
        most_common_element = counter.most_common(1)[0][0]
        
        # 将出现次数最多的元素添加到结果列表中
        result.append(most_common_element)
    
    return result

class Trainer:
    def __init__(self, args):
        
        
        # 修改数据加载部分
        train_df, test_df, self.label_to_idx, self.idx_to_label = load_csv(args.train_csv, args.test_csv)

        # 划分训练集和验证集
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

        train_df.to_csv('/root/lbs/LDB/dataset/train_split.csv', index=False)

        
        # 创建数据集
        train_dataset = ButterflyDataset(train_df, args.train_images, self.label_to_idx, datatype = 'train',augment_times= args.augment_times,if_double=True)
        val_dataset = ButterflyDataset(val_df, args.train_images, self.label_to_idx, datatype='val')
        test_dataset = ButterflyDataset(test_df, args.test_images)

        #train_dataset1 = ButterflyDataset(train_df, args.train_images, self.label_to_idx, datatype = 'train',augment_times= args.augment_times, if_blackandwhite=True)
        #val_dataset1 = ButterflyDataset(val_df, args.train_images, self.label_to_idx, datatype='val', if_blackandwhite=True)
        #test_dataset1 = ButterflyDataset(test_df, args.test_images, if_blackandwhite=True)
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print('Dataloader created!')
        
        #模型实例化与损失函数、优化函数定义
        self.criterion = nn.CrossEntropyLoss()
        #self.model = CurrentModel(num_classes=len(self.label_to_idx)).to(args.device)
        
        self.model1 = create_model(
        'unireplknet_b', 
        pretrained=False, 
        num_classes=len(self.label_to_idx), 
        drop_path_rate=0.2,
        layer_scale_init_value=1e-6,
        head_init_scale=0.001,
        )
        
        self.model2 = create_model(
                'MetaFG_2',
                pretrained=False,
                num_classes=len(self.label_to_idx), 
                drop_path_rate=0.1,
                img_size=224,
                only_last_cls=False,
                extra_token_num=1,
                meta_dims=[]
        )
        
        self.model3 = ViT(num_classes=len(self.label_to_idx)).to(args.device)

        self.model4 = build_models(config, 75)
        
        self.model5 = create_model(
        'convnext_large', 
        
        )
        
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for images, labels in loader:
            images, labels = images.to(args.device), labels.to(args.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(loader)
        
        return avg_loss, acc, balanced_acc

    
        
       
    @torch.no_grad()
    def predict(self):
        # 加载最佳模型
        

        self.model.load_state_dict(torch.load(self.model_dir + 'best_model.pth'))
            
        self.model.eval()
        
        
        predictions = []
        for images, img_names in tqdm(self.test_loader, desc="Predicting"):
            images = images.to(args.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [self.idx_to_label[idx.item()] for idx in predicted]
            predictions.extend(zip(img_names, predicted_labels))
            
        return predictions

    # 保存预测结果，生成 submission.csv
    def save_predictions(self, predictions):
        submission_df = pd.DataFrame(predictions, columns=['filename', 'label'])
        submission_df.to_csv(self.submission_file, index=False)
        print(f'submission saved at {self.submission_file}')
        
    @torch.no_grad()
    def evaluate_with_weights(self, weights):
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc="Evaluating"):
            images, labels = images.to(args.device), labels.to(args.device)
            
            prob = torch.zeros_like(nn.functional.softmax(self.models[0](images), dim=1))
            
            for i in range(len(self.models)):
                outputs = self.models[i](images)
                prob += weights[i] * nn.functional.softmax(outputs, dim=1)
            
            avg_prob = prob / sum(weights)
            _, predicted = torch.max(avg_prob, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        print(f'weight: {weights} acc: {acc:.10f} b_acc: {balanced_acc:.10f}')
        return balanced_acc 
     
     
        
    @torch.no_grad()
    def multi_predict(self):
        # 加载最佳模型

        self.models = []

        self.model1.load_state_dict(torch.load('/root/lbs/LDB/logs/UniRepLKNet_blend/02061555_best/best_model.pth'))
        self.model2.load_state_dict(torch.load('/root/lbs/LDB/logs/MetaFormer/02062248/best_model.pth'))
        self.model3.load_state_dict(torch.load('/root/lbs/LDB/logs/baseline/02062210/best_model.pth'))
        
        checkpoint = torch.load('/root/lbs/LDB/models/MPSA/output/cub/Ours 02-06_23-38/checkpoint.bin', map_location='cpu')
        state_dicts = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}
        state_dicts = {k.replace('_orig_mod.', ''): v for k, v in state_dicts.items()}
        msg = self.model4.load_state_dict(state_dicts, strict=True)
        self.model5.load_state_dict(torch.load('/root/lbs/LDB/logs/ConvNext_L/02081245/best_model.pth'))
        
        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()
        self.model5.eval()
        
        self.models.append(self.model1)
        self.models.append(self.model3)
        self.models.append(self.model2)
        self.models.append(self.model4)
        self.models.append(self.model5)
        
        for model in self.models:
            model.to(args.device)

        
        
        print('start eval')
        
        best_weights = None
        best_balanced_acc = 0
        weight_range = np.arange(3, 17, 1) 
        '''for weights in itertools.product(weight_range, repeat=5):  # 更改范围为 range(10)
            if weights == (10,10,10,10,10) :
                
                balanced_acc = self.evaluate_with_weights(weights)
                if balanced_acc > best_balanced_acc:
                    best_balanced_acc = balanced_acc
                    best_weights = weights'''
        best_weights = (0.25, 0.05, 0.15, 0.35, 0.20)
        #best_weights = (1, 1, 1, 1, 1)
        balanced_acc = self.evaluate_with_weights(best_weights)
        
        print(f'Best weights: {best_weights}, Best balanced accuracy: {best_balanced_acc}')
        
        val_acc_int = int(balanced_acc * 10000) 

        self.submission_file = f'./submission/sub_UniRe+MetaFG+Vit+MPSA+Conv_{val_acc_int}.csv'
        print(val_acc_int)
        
        predictions = []
        for images, img_names in tqdm(self.test_loader, desc="Predicting"):
            images = images.to(args.device)
            prob = torch.zeros_like(nn.functional.softmax(self.models[0](images), dim=1))
            for model in range(5):
                outputs = self.models[model](images)
                prob += best_weights[model] * nn.functional.softmax(outputs, dim=1)
            avg_prob = prob / len(self.models)
            _, predicted = torch.max(avg_prob, 1)
            predicted_labels = [self.idx_to_label[idx.item()] for idx in predicted]
            predictions.extend(zip(img_names, predicted_labels))
            
        return predictions

    # 保存预测结果，生成 submission.csv

if __name__ == '__main__':
    args = config_args()
    set_seed(args.seed)
    
    trainer = Trainer(args)
    
    predictions = trainer.multi_predict()
    trainer.save_predictions(predictions)