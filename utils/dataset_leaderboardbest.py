import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from rembg import remove
import torch
import random
import shutil


#使用叠照片推理

'''dataset_path = '/root/lbs/LDB/dataset'  #蝴蝶数据集
train_csv = os.path.join(dataset_path, 'train.csv') 
test_csv = os.path.join(dataset_path, 'test.csv')
train_images = os.path.join(dataset_path, 'train_images')
test_images = os.path.join(dataset_path, 'test_images')
submission_file = '/root/lbs/LDB/submission.csv'  #这个不用改了

batch_size = 32
epoch_num = 60
learning_rate = 1e-5
devices = 'cuda'
random_seed = 114514
cuda_device = '1,2,3,4,5,6,7'
augment_times = 4'''


'''self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.8)
        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=args.learning_rate, alpha=0.7)


        steps_per_epoch = len(self.train_loader)
        total_steps = args.epochs * steps_per_epoch
        
        
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
'''






def overlay_random_images(src_folder, num_files=2, alpha=0.5):
    # 获取源文件夹中的所有文件
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 随机选取指定个数的文件
    selected_files = random.sample(files, num_files)

    # 加载第一张图片
    base_img = Image.open(os.path.join(src_folder, selected_files[0])).convert('RGBA')

    # 依次加载并叠加剩余的图片
    for file in selected_files[1:]:
        img = Image.open(os.path.join(src_folder, file)).convert('RGBA')
        img = img.resize(base_img.size)  # 调整图片大小一致
        base_img = Image.blend(base_img, img, alpha=alpha)

    # 转换为RGB图像
    blended_rgb = base_img.convert('RGB')

    return blended_rgb

def organize_files_by_class(csv_file, src_folder, dst_folder):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 遍历每一行
    for _, row in df.iterrows():
        file_name = row['filename']
        file_class = row['label']

        # 构建源文件路径和目标文件夹路径
        src_path = os.path.join(src_folder, file_name)
        class_folder = os.path.join(dst_folder, str(file_class))

        # 确保目标子文件夹存在
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # 构建目标文件路径
        dst_path = os.path.join(class_folder, file_name)

        # 复制文件到目标文件夹
        shutil.copy2(src_path, dst_path)

transform_normal= transforms.Compose([
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小，增加尺度变化
    transforms.RandomHorizontalFlip(p=0.25),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.25),  # 随机垂直翻转
    
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.RandomPerspective(distortion_scale=0.2, p=0.25),  # 随机透视变换
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # 高斯模糊
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    #
])
transform_plus = transforms.Compose([
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小，增加尺度变化
    transforms.RandomHorizontalFlip(p=0.25),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.1),  # 随机垂直翻转
      # 颜色抖动
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomRotation(degrees=15),  # 随机旋转

    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # 随机透视变换
    transforms.RandomGrayscale(p=0.05),  # 随机灰度化
    
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊
    transforms.ToTensor(),  # 转换为张量
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.4), ratio=(0.3, 3.3), value='random'), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    
    #
])

transform_mask = transforms.Compose([
    #transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪并调整大小，增加尺度变化
    transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
    transforms.RandomVerticalFlip(p=0.2),  # 随机垂直翻转
      # 颜色抖动
    transforms.RandomRotation(degrees=45),  # 随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # 随机透视变换
    #transforms.RandomGrayscale(p=0.1),  # 随机灰度化
    
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 高斯模糊
    transforms.ToTensor(),  # 转换为张量
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.5), ratio=(0.3, 3.3), value=(0,0,0)), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    
    #
])
transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ButterflyDataset(Dataset):
    def __init__(self, dataframe, image_dir, label_to_idx=None, datatype='val', augment_times=1, if_blackandwhite=False, if_double=False):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.label_to_idx = label_to_idx
        self.datatype = datatype
        self.augment_times = augment_times  # 数据增强的次数
        self.color = if_blackandwhite
        self.double = if_double
        if datatype == 'train':
            self.dataframe.to_csv('./dataset/tmp_train_split.csv')
            if os.path.exists('./dataset/tmp_train_class'):
                shutil.rmtree('./dataset/tmp_train_class')
            organize_files_by_class('./dataset/tmp_train_split.csv', './dataset/train_images', './dataset/tmp_train_class')
    def __len__(self):
        if self.datatype == 'train':
            return len(self.dataframe) * self.augment_times
        else:
            return len(self.dataframe)

    def __getitem__(self, idx):
        if self.datatype == 'train':
            original_idx = idx // self.augment_times
        else:
            original_idx = idx

        img_name = self.dataframe.iloc[original_idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        
        if self.color:
            image = Image.open(img_path).convert('L')
            image = Image.merge("RGB", (image, image, image))
        else:
            
            image = Image.open(img_path).convert('RGB')

        
        
        if self.datatype == 'train':
            image = transform_plus(image)
        else:
            image = transform_val_test(image)
            
        
        if self.label_to_idx:
            label = self.dataframe.iloc[original_idx, 1]
            label = self.label_to_idx[label]   
            
            if self.datatype == 'train' and self.double:
                if random.random() < 0.5:
                    src_folder = './dataset/tmp_train_class/' + self.dataframe.iloc[original_idx, 1]
                    
                    image = overlay_random_images(src_folder, num_files=2)
                    image = transform_normal(image)
                    
            return image, label
        else:
            return image, img_name
        
        
        
