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
from collections import deque
from models.MetaFG.MetaFG import *
from models.MetaFG.MetaFG_meta import *
from models.MetaFormer import load_pretained
from models.MPSA.setup import config
from models.MPSA.models.build import build_models

import models.convnext
import models.convnext_isotropic
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model

from collections import Counter
import sys
import os
import contextlib
from collections import defaultdict
import warnings

# 抑制所有警告
warnings.filterwarnings("ignore")

'''class FeatureFusionModule(nn.Module):
    def __init__(self, feat_dims=[1920, 1024], num_classes=75):
        super().__init__()
        # 自适应特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(sum(feat_dims), 512, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.classifier = nn.Linear(512, num_classes)  # 需替换num_classes

    def forward(self, features):
        fused = self.fusion(torch.cat(features, dim=1))
        return self.classifier(fused)'''
    
class FeatureFusionModule(nn.Module):
    def __init__(self, mpsa_dim=1920, unirep_dim=1024, num_classes=1920):
        super().__init__()
        # 通道对齐层
        self.mpsa_conv = nn.Conv2d(mpsa_dim, 128, 1)
        self.unirep_conv = nn.Conv2d(unirep_dim, 128, 1)
        
        # 注意力融合模块
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 256, 1),
            nn.Sigmoid()
        )
        
        # 多尺度增强
        self.pyramid_pool = nn.ModuleList([
            nn.Conv2d(256, 256, 3, dilation=d, padding=d)
            for d in [1, 2, 4]
        ])
        
        self.head = nn.Linear(768, num_classes)
        
        
    def forward(self, feat_list):
        # 特征对齐
        for feat in range(2):
            feat_list[feat] = feat_list[feat].unsqueeze(-1).unsqueeze(-1)
        mpsa_feat = self.mpsa_conv(feat_list[0])  
        unirep_feat = self.unirep_conv(feat_list[1]) 
        
        # 拼接特征
        fused = torch.cat([mpsa_feat, unirep_feat], dim=1) 
        
        # 通道注意力
        channel_weights = self.channel_att(fused)
        fused = fused * channel_weights
        
        # 多尺度融合
        pyramid_feats = [pool(fused) for pool in self.pyramid_pool]
        final_feat = torch.cat(pyramid_feats, dim=1)  # [B,768,H,W]
        final_feat = final_feat.squeeze(-1).squeeze(-1)
        final_feat = self.head(final_feat)
        
        
        return final_feat

class DynamicWeightController:
    def __init__(self, model_names, init_weights=[], 
                trend_window=5, decay=0.9, min_weight=0.05):
        """
        :param model_names: 模型名称列表
        :param init_weights: 初始权重列表
        :param trend_window: 趋势计算窗口大小
        :param decay: 动量衰减系数
        :param min_weight: 最小权重阈值
        """
        assert len(model_names) == len(init_weights)
        assert abs(sum(init_weights) - 1.0) < 1e-6
        
        self.model_names = model_names
        self.weights = dict(zip(model_names, init_weights))
        self.min_weight = min_weight
        self.decay = decay
        
        # 历史记录
        self.acc_history = {name: deque(maxlen=trend_window) for name in model_names}
        self.ensemble_acc = deque(maxlen=trend_window)
        
    def update(self, model_acc, ensemble_acc):
        """
        :param model_acc: 各模型本epoch验证准确率（dict）
        :param ensemble_acc: 集成模型本epoch准确率（float）
        """
        # 1. 记录历史数据
        for name in self.model_names:
            self.acc_history[name].append(model_acc[name])
        self.ensemble_acc.append(ensemble_acc)
        self._detect_anomalies()
        # 2. 计算各模型趋势得分
        self.prev_weights = self.weights.copy()
        trend_scores = self._compute_trend_scores()
        
        # 3. 计算集成趋势得分
        ensemble_trend = self._compute_ensemble_trend()
        
        # 4. 计算权重调整量
        delta_weights = self._compute_delta_weights(trend_scores, ensemble_trend)
        
        # 5. 应用权重调整
        self._apply_updates(delta_weights)
        
        self._update_decay_factor()
        
        return self.weights.copy()
    
    def _compute_trend_scores(self):
        """计算各模型的趋势得分（加权移动平均）"""
        scores = {}
        for name in self.model_names:
            history = list(self.acc_history[name])
            if len(history) < 2:
                scores[name] = 0.0
                continue
                
            # 计算指数加权趋势
            weights = [self.decay**i for i in range(len(history))]
            weights = np.array(weights[::-1]) / sum(weights)
            diff = np.diff(history)
            scores[name] = np.dot(diff, weights[:-1])
            
        return scores
    
    def _compute_ensemble_trend(self):
        """计算集成准确率趋势"""
        if len(self.ensemble_acc) < 2:
            return 0.0
            
        recent = np.mean(list(self.ensemble_acc)[-2:])
        baseline = np.mean(list(self.ensemble_acc)[:-2] or [recent])
        return recent - baseline
        
    def _compute_delta_weights(self, trend_scores, ensemble_trend):
        """综合计算权重调整量"""
        deltas = {}
        total_trend = sum(abs(ts) for ts in trend_scores.values())
        
        for name in self.model_names:
            # 个体趋势贡献
            individual = trend_scores[name] / (total_trend + 1e-6)
            
            # 集成趋势适配
            if ensemble_trend > 0:
                # 整体提升时强化优势模型
                ensemble_factor = max(0, individual)
            else:
                # 整体下降时抑制弱势模型
                ensemble_factor = min(0, individual)
                
            # 当前权重适配（防止马太效应）
            weight_factor = 1 - np.tanh(self.weights[name] * 3)  # 抑制过高权重
            
            # 综合调整量
            delta = individual * ensemble_factor * weight_factor
            deltas[name] = delta
            
        # 标准化调整量
        max_delta = max(abs(d) for d in deltas.values()) or 1.0
        return {k: v/max_delta * 0.1 for k,v in deltas.items()}  # 限制最大调整幅度
    
    def _apply_updates(self, delta_weights):
        """应用权重更新"""
        # 1. 初步更新
        for name in self.model_names:
            self.weights[name] += delta_weights[name]
            
        # 2. 应用边界约束
        for name in self.model_names:
            self.weights[name] = max(self.weights[name], self.min_weight)
            
        # 3. 归一化
        total = sum(self.weights.values())
        self.weights = {k: v/total for k,v in self.weights.items()}

    def _detect_anomalies(self):
        """检测异常权重变化"""
        current_acc = np.mean(list(self.ensemble_acc))
        if len(self.ensemble_acc) > 3:
            if current_acc < np.percentile(list(self.ensemble_acc)[:-1], 10):
                # 集成准确率骤降时重置部分权重
                for name in self.model_names:
                    if self.weights[name] < 0.1:
                        self.weights[name] = min(0.1, self.weights[name]*1.5)
    
    def _update_decay_factor(self):
        """根据权重稳定性自动调整衰减速率"""
        weight_changes = [abs(self.weights[name] - self.prev_weights[name]) 
                        for name in self.model_names]
        avg_change = np.mean(weight_changes)
        self.decay = max(0.7, min(0.95, 1 - avg_change*2))
    
    def _to_scalar(self, value):
        """将各种类型输入转换为Python浮点数"""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        elif isinstance(value, np.ndarray):
            return float(value.item())
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(f"Unsupported accuracy type: {type(value)}")

class FusionSystem:
    def __init__(self, models, num_classes, freeze_backbones=True, model_dir=''):
        """
        :param models: 包含五个预训练模型的字典，格式：
           {'MPSA': model1, 'UniRepLKNet': model2, ...}
        """
        # 注册特征提取钩子
        
        # 初始化融合模块
        self.fusion = FeatureFusionModule(num_classes=75).cuda()
        self.controller = DynamicWeightController(
            model_names=['MPSA', 'ConVNeXt', 'UniRepLKNet', 'MetaFormer', 'VisionTransformer'] + ['Fusion'],
            init_weights= [0.20, 0.24, 0.17, 0.15, 0.12, 0.12] # 初始权重分配
        )
        self.models = models
        self.criterion = nn.CrossEntropyLoss()
        torch.use_deterministic_algorithms(False)
        for name, model in models.items():
            
            for param in model.parameters():
                param.requires_grad_(False)
        self.model_dir = model_dir
        self.log_dir = model_dir + 'log.txt'
        print('train ready!')
        
    def train_fusion(self, train_loader, val_loader, epochs=30):
        optimizer = torch.optim.AdamW(self.fusion.parameters(), lr=1e-4)
        scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # 最大学习率
        total_steps=4800,  # 总步数
        epochs=120,  # 总训练轮数
        pct_start=0.2,  # 20%的时间用于热身
        anneal_strategy='cos',  # 余弦退火
        div_factor=25,  # 初始学习率 = max_lr / 25
        final_div_factor=1e4  # 最终学习率 = max_lr / (25 * 1e4)
        )
        for epoch in range(epochs):
            # 训练阶段
            self._run_epoch(train_loader, optimizer, training=True)
            
            # 验证阶段
            val_acc = self._run_epoch(val_loader, None, training=False)
            self.controller.update(val_acc)
            
            print(f"Epoch {epoch+1}: Fusion Acc {val_acc['Fusion']:.3f} | "
                  f"Weights {self.controller.weights}")

    def _run_epoch(self, loader, optimizer, scheduler, training=True):
        total_acc = defaultdict(float)
        self.models = {k: v.train() for k,v in self.models.items()}
        for images, labels in tqdm(loader):
            images = images.cuda()
            labels = labels.cuda()
            # 前向传播获取特征
                # 并行模型推理
            outputs = {}
            feats = {}
            for name, model in self.models.items():
                if name in ['MPSA', 'UniRepLKNet']:
                    outputs[name], feats[name] = model(images)
                else:
                    outputs[name] = model(images)
            
            # 融合模型输出
            fusion_feats = [feats['MPSA'], feats['UniRepLKNet']]

            outputs['Fusion'] = self.fusion(fusion_feats)
            
            # 反向传播（仅训练模式）
            if training and optimizer:
                loss = self.criterion(outputs['Fusion'], labels) 
                       
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        # 计算平均准确率
        return loss

    def predict(self, image):
        """推理接口"""
        with torch.no_grad():
            # 获取各模型预测
            outputs = {}
            for name, model in self.models.items():
                outputs[name] = model(image)
            
            # 获取融合预测
            fusion_feats = [self.feature_maps[name] for name in self.models]
            outputs['Fusion'] = self.fusion(fusion_feats)
            
            # 加权综合预测
            final_logits = sum(
                outputs[name] * self.controller.weights[name]
                for name in outputs
            )
            return final_logits.argmax(dim=1)
        
class EnhancedFusionSystem(FusionSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_acc = 0.0
        self.best_weights = None
        

    def validate(self, val_loader, verbose=False):
        """增强验证函数"""
        self.models = {k: v.eval() for k,v in self.models.items()}
        self.fusion.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        details = defaultdict(list)
        resultdict = {
            'MPSA': [],
            'UniRepLKNet': [],
            'ConVNeXt': [],
            'VisionTransformer': [],
            'MetaFormer': [],
            'Fusion': []
        }
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating"):
                images = images.cuda()
                labels = labels.cuda()
                
                # 获取各模型输出
                outputs = {}
                feats = {}
                for name, model in self.models.items():
                    if name in ['MPSA', 'UniRepLKNet']:
                        outputs[name], feats[name] = model(images)
                    else:
                        outputs[name] = model(images)
                # 融合模型输出
                fusion_feats = [feats['MPSA'], feats['UniRepLKNet']]
                outputs['Fusion'] = self.fusion(fusion_feats)
                
                # 计算加权损失
                loss = sum(nn.CrossEntropyLoss()(outputs[name], labels) 
                         * self.controller.weights.get(name, 1.0) 
                         for name in outputs)
                total_loss += loss.item()
                
                
                
                for k,v in outputs.items():
                    _, pre = torch.max(v, 1)
                    
                    resultdict[k].extend(pre.cpu().numpy())
                
                # 收集预测结果
                final_preds = self._ensemble_predict(outputs)
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                
                # 记录详细预测
                if verbose:
                    for name in self.models:
                        preds = outputs[name].argmax(dim=1)
                        details[name].extend(zip(
                            labels.cpu().numpy(),
                            preds.cpu().numpy()
                        ))
        accdict = dict()
        for k,v in resultdict.items():
            accdict[k] = np.mean(np.array(v) == np.array(all_labels))
            
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        # 打印样本级结果
        if verbose:
            print("\nSample-level Validation Results:")
            for idx in range(min(5, len(all_labels))):
                print(f"Sample {idx+1}: True={all_labels[idx]}, Pred={all_preds[idx]}")
            print(f"\nValidation Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
        
        # 更新最佳模型
        if acc > self.best_acc:
            self.best_acc = acc
            self._save_best_model()
        
        return avg_loss,  acc, balanced_acc, accdict

    def _ensemble_predict(self, outputs):
        """加权集成预测"""
        weighted_logits = sum(
            outputs[name] * self.controller.weights[name] 
            for name in outputs
        )
        ijk, predicted = torch.max(weighted_logits, 1)
        return predicted

    def _save_best_model(self):
        """保存最佳模型状态"""
        torch.save({
            'fusion_state': self.fusion.state_dict(),
            'weights': self.controller.weights,
            'best_acc': self.best_acc
        }, f"{self.model_dir}best_fusion_model.pth")

    def load_best_model(self):
        """加载最佳模型"""
        checkpoint = torch.load(f"{self.model_dir}best_fusion_model.pth")
        self.fusion.load_state_dict(checkpoint['fusion_state'])
        self.controller.weights = checkpoint['weights']
        self.best_acc = checkpoint['best_acc']

    def train_fusion(self, train_loader, val_loader, epochs=30):
        optimizer = torch.optim.AdamW([
            {'params': self.fusion.parameters(), 'lr': 1e-4},
            {'params': [v for m in self.models.values() 
                       for v in m.parameters()], 'lr': 1e-7}  # 可选微调
        ])
        scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,  # 最大学习率
        total_steps=4800,  # 总步数
        epochs=120,  # 总训练轮数
        pct_start=0.2,  # 20%的时间用于热身
        anneal_strategy='cos',  # 余弦退火
        div_factor=25,  # 初始学习率 = max_lr / 25
        final_div_factor=1e4  # 最终学习率 = max_lr / (25 * 1e4)
        )
        for epoch in range(epochs):
            # 训练阶段
            self._run_epoch(train_loader, optimizer, scheduler, training=True)
            
            # 验证阶段
            avg_loss, val_acc, val_bal_acc, accdict = self.validate(val_loader)
            self.controller.update(accdict, val_acc)
            facc = accdict['Fusion']
            print(f"Epoch {epoch+1}: Fusion Acc {val_acc:.4f}, {val_bal_acc:.4f} | "
                  f"Weights {self.controller.weights}")
            print(f'Fusion acc {facc:.4f}')
            with open(self.log_dir, 'a+') as log:
                log.write(f'Epoch {epoch+1}: \n')
                log.write(f"Val   - weight: {self.controller.weights}, Acc: {val_acc:.4f}, Balanced Acc: {val_bal_acc:.4f}\n")
        return self.best_acc
    def predict(self, test_loader, idx_to_label):
        """增强推理函数"""
        self.load_best_model()
        self.models = {k: v.eval() for k,v in self.models.items()}
        self.fusion.eval()
        
        all_preds = []
        with torch.no_grad():
            for images, img_names in tqdm(test_loader, desc="Predicting"):
                # 支持不同数据格式 (images, labels) 或 (images, metadata)
                images = images.to(args.device)
                
                # 获取各模型输出
                outputs = dict()
                feats = dict()
                for name, model in self.models.items():
                    if name in ['MPSA', 'UniRepLKNet']:
                        outputs[name], feats[name] = model(images)
                    else:
                        outputs[name] = model(images)
                # 融合模型输出
                fusion_feats = [feats['MPSA'], feats['UniRepLKNet']]
                outputs['Fusion'] = self.fusion(fusion_feats)
                
                
                # 集成预测
                batch_preds = self._ensemble_predict(outputs)
                predicted_labels = [idx_to_label[idx.item()] for idx in batch_preds]
                all_preds.extend(zip(img_names, predicted_labels))
        
        return all_preds
       
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr



class Trainer:
    def __init__(self, args):
        
        with suppress_output():
            train_df, test_df, self.label_to_idx, self.idx_to_label = load_csv(args.train_csv, args.test_csv)

            # 划分训练集和验证集
            train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=114514, stratify=train_df['label'])

            val_df, val_multi_df = train_test_split(val_df, test_size=0.4, random_state=114514, stratify=val_df['label'])

            train_df.to_csv('/root/lbs/LDB/dataset/train_split.csv', index=False)

            
            # 创建数据集
            train_dataset = ButterflyDataset(train_df, args.train_images, self.label_to_idx, datatype = 'train',augment_times=2,if_double=True)
            val_dataset = ButterflyDataset(val_multi_df, args.train_images, self.label_to_idx, datatype='val')
            test_dataset = ButterflyDataset(test_df, args.test_images)

            #train_dataset1 = ButterflyDataset(train_df, args.train_images, self.label_to_idx, datatype = 'train',augment_times= args.augment_times, if_blackandwhite=True)
            #val_dataset1 = ButterflyDataset(val_df, args.train_images, self.label_to_idx, datatype='val', if_blackandwhite=True)
            #test_dataset1 = ButterflyDataset(test_df, args.test_images, if_blackandwhite=True)
            
            # 创建数据加载器
            self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
            self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        print('Dataloader created!')
            

        with suppress_output():    
            self.model1 = create_model(
            'unireplknet_b', 
            pretrained=False, 
            num_classes=len(self.label_to_idx), 
            drop_path_rate=0.2,
            layer_scale_init_value=1e-6,
            head_init_scale=0.001,
            if_feat=True
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

            self.model4 = build_models(config, 75, if_feat=True)
            
            self.model5 = create_model(
            'convnext_large', 
            
            )
        
        print('model loaded!')
        
        with suppress_output():
            self.trained_models = []

            self.model1.load_state_dict(torch.load('/root/lbs/LDB/logs/UniRepLKNet_blend/02150434/best_model.pth'))
            self.model2.load_state_dict(torch.load('/root/lbs/LDB/logs/MetaFormer/02150435/best_model.pth'))
            self.model3.load_state_dict(torch.load('/root/lbs/LDB/logs/baseline/02150434/best_model.pth'))
            
            self.model4.load_state_dict(torch.load('/root/lbs/LDB/logs/MPSA/02150434/best_model.pth'))
            self.model5.load_state_dict(torch.load('/root/lbs/LDB/logs/ConvNext_L/02150435/best_model.pth'))
            
            
            self.trained_models.append(self.model1)
            self.trained_models.append(self.model3)
            self.trained_models.append(self.model2)
            self.trained_models.append(self.model4)
            self.trained_models.append(self.model5)
            
            for model in self.trained_models:
                model.to(args.device)
            
        
        
        self.models={
            'MPSA': self.trained_models[3],
            'UniRepLKNet': self.trained_models[0],
            'ConVNeXt': self.trained_models[4],
            'VisionTransformer': self.trained_models[2],
            'MetaFormer': self.trained_models[1]
        }
        print('checkpoint loaded!')
        
        
        
        self.main_log_dir = './logs/'
        self.model_dir = self.main_log_dir + 'fusion' + '/' + datetime.now().strftime('%m%d%H%M') + '/'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.log_dir = self.model_dir + 'log.txt'
        
        self.submission_file = './submission/' + 'fusion' + '_' + datetime.now().strftime('%m%d%H%M')
        
        print('log loaded')
        
    def train(self, epoch=30):
        fusion_system = EnhancedFusionSystem(models=self.models, num_classes=75, freeze_backbones=False, model_dir=self.model_dir)
        best_acc = fusion_system.train_fusion(self.train_loader, self.val_loader, epochs=epoch)
        self.submission_file += '_' + f'{best_acc:.4f}' + '.csv'
        predictions = fusion_system.predict(self.test_loader, self.idx_to_label)
        
        return predictions
    
    def save_predictions(self, predictions):
        submission_df = pd.DataFrame(predictions, columns=['filename', 'label'])
        submission_df.to_csv(self.submission_file, index=False)
        print(f'submission saved at {self.submission_file}')
if __name__ == '__main__':
    args = config_args()
    set_seed(args.seed)
    
    trainer = Trainer(args)
    
    
    predictions = trainer.train(epoch=120)
    trainer.save_predictions(predictions)