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
import timm
from transformers import AutoModelForImageClassification, AutoModel
#!export HF_ENDPOINT=https://hf-mirror.com
from datetime import datetime
import time
from utils.utils import set_seed, load_csv
from utils.config import config_args
from utils.dataset import ButterflyDataset
from models.MetaFG.MetaFG import *
from models.MetaFG.MetaFG_meta import *
from models.MetaFormer import load_pretained
from torch.optim.lr_scheduler import OneCycleLR
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy
class Trainer:
    def __init__(self, args):
        
        
        # 修改数据加载部分
        train_df, test_df, self.label_to_idx, self.idx_to_label = load_csv(args.train_csv, args.test_csv)

        # 划分训练集和验证集
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['label'])

        # 创建数据集
        train_dataset = ButterflyDataset(train_df, args.train_images, self.label_to_idx, datatype = 'train',augment_times= args.augment_times, if_double=True)
        val_dataset = ButterflyDataset(val_df, args.train_images, self.label_to_idx, datatype='val')
        test_dataset = ButterflyDataset(test_df, args.test_images)

        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        print('Dataloader created!')
        
        #模型实例化与损失函数、优化函数定义
        
        #self.model = CurrentModel(num_classes=len(self.label_to_idx)).to(args.device)
        self.criterion = nn.CrossEntropyLoss()
        self.model = create_model(
                'MetaFG_2',
                pretrained=False,
                num_classes=len(self.label_to_idx), 
                drop_path_rate=0.1,
                img_size=224,
                only_last_cls=False,
                extra_token_num=1,
                meta_dims=[]
        )
        
        
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        steps_per_epoch = len(self.train_loader)
        total_steps = args.epochs * steps_per_epoch
        # 加载MetaF权重
        self.scheduler = OneCycleLR(
        self.optimizer,
        max_lr=0.0001,  # 最大学习率
        total_steps=total_steps,  # 总步数
        epochs=args.epochs,  # 总训练轮数
        pct_start=0.1,  # 20%的时间用于热身
        anneal_strategy='cos',  # 余弦退火
        div_factor=25,  # 初始学习率 = max_lr / 25
        final_div_factor=1e4  # 最终学习率 = max_lr / (25 * 1e4)
        )
        
        #load_pretained(self.model)
        checkpoint = torch.load('/root/lbs/LDB/models/metafg_2_21k_224.pth', map_location='cpu')
        
        state_dict = self.model.state_dict()
        for key in ['head.weight', 'head.bias']:
            if key in checkpoint and checkpoint[key].shape != state_dict[key].shape:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint[key]
        self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.to(args.device)
        
        
        self.model.name = 'MetaFormer'
        
        print('model loaded!')
        
        
        # 添加日志
        self.main_log_dir = './logs/'
        self.model_dir = self.main_log_dir + self.model.name + '/' + datetime.now().strftime('%m%d%H%M') + '/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.log_dir = self.model_dir + 'log.txt'
        
        self.submission_file = './submission/' + self.model.name + '_' + datetime.now().strftime('%m%d%H%M')
        
        
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

    def train(self):
        best_val_acc = 0
        for epoch in range(args.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for images, labels in progress_bar:
                images, labels = images.to(args.device), labels.to(args.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix(loss=train_loss / len(progress_bar))
            
            # 计算训练集指标
            train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
            train_balanced_acc = balanced_accuracy_score(train_labels, train_preds)
            train_loss = train_loss / len(self.train_loader)
            
            # 验证阶段
            val_loss, val_acc, val_balanced_acc = self.evaluate(self.val_loader)
            
            # 打印训练和验证指标
            print(f"\nEpoch {epoch+1}/{args.epochs}:")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Balanced Acc: {train_balanced_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Balanced Acc: {val_balanced_acc:.4f}")
            
            with open(self.log_dir, 'a+') as log:
                log.write(f'Epoch {epoch+1}: \n')
                log.write(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Balanced Acc: {train_balanced_acc:.4f}\n")
                log.write(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Balanced Acc: {val_balanced_acc:.4f}\n")

            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_dir + 'best_model.pth')
            
            with open(self.log_dir, 'a+') as log:
                log.write(f'Best val acc: {val_acc}')
        val_acc_int = int(best_val_acc * 10000) 
        self.submission_file = self.submission_file + '_' + str(val_acc_int) + '.csv'
       
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

if __name__ == '__main__':
    args = config_args()
    set_seed(args.seed)
    
    trainer = Trainer(args)
    trainer.train()
    
    predictions = trainer.predict()
    trainer.save_predictions(predictions)