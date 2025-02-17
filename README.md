# 上科大 ‘慧炼丹心杯’ CUDA Out of Memory队代码备份

## 写在前面  
CUDA Out Of Memory队是一支由24届新生组成的队伍，我们在模型设计、代码管理上可能有诸多的不成熟之处。  
~~没有随时保存超参数 忘记检查其中一次训练的随机种子 项目中满屏的本地路径~~  
请大家在跑本文件遇到问题时多多包涵，我们会及时检修  
~~屎山守护精灵尝试发力~~  

### 我们借用的开源模型（包括他们提供的ImageNet上预训练权重）
**VisionTransformer**<https://github.com/google-research/vision_transformer>  
**UniRepLKNet**<https://github.com/AILab-CVC/UniRepLKNet>  
**ConvNeXt**<https://github.com/facebookresearch/ConvNeXt>  
**MPSA**<https://github.com/mobulan/MPSA>  
**MetaFormer**<https://github.com/dqshuai/metaformer>  
### 我们尝试过但没有放进最终版本的开源模型
**IELT**<https://github.com/mobulan/IELT>   
### 没有跑通但参考过的工作  
**CMAL**<https://github.com/Dichao-Liu/CMAL>  
**FGVC-PIM**<https://github.com/Dichao-Liu/CMAL>

***

## 开始  
```
git clone https://github.com/negativegluon/COM.git
cd COM
```

### 安装环境  
我们本地程序在python3.8上运行。CUDA版本是12.2.  
```
conda create -n ldb-com python==3.8.0
```
```
conda activate ldb-com
pip install -r requirements.txt
```
### 下载预训练权重（有可能仅推理也需要这些操作）
UniRepLKNet: 从上文提及的UniRepLKNet公开仓库> ImageNet-22K Pretrained Weights下载UniRepLKNet-B  
MetaFormer: 从上文提及的MetaFormer公开仓库> model zoo下载 MetaFormer-2， 224x224， 21k model   
ConvNeXt：从上文提及的ConvNeXt公开仓库> ImageNet-22K trained models下载ConvNeXt-L，224x224，22k model  
不要更改以上三个文件的文件名，将其移动至/models  

VisionTransformer：
```
git clone https://hf-mirror.com/google/vit-large-patch16-224
```
安装该文件后，将models/baseline中modelname引用的文件夹地址更换为这个项目在你本地的地址。  
MPSA：在models/MPSA中，按照MPSA公开仓库中Training部分操作  

### 下载数据集  
从“慧炼丹心杯”比赛界面中下载比赛数据集，并放置在/dataset文件夹中。  
文件结构应当形如：
>dataset  
>├── train_images  
>│   ├── Image_xxx.jpg  
>├── test_images  
>│   ├── Image_xxx.jpg  
>├── train.csv  
>└── test.csv  
### 下载我们所训练的权重（如果你不需要推理我们的模型，可跳过） 
访问<https://epan.shanghaitech.edu.cn/l/IFD6ag> (提取码：edhy)  
（我们设置了需要网盘登陆才能下载文件）  
将logs文件夹移动至项目主文件夹下。  
如果你需要现在推理，打开multieval.py并在multi_predict函数中参考原先地址，将model1至5所加载的文件地址分别更换为你的本地地址。  

***

## 训练  
将utils/dataset_leaderboardbest.py内容覆盖至utils/dataset中  
查看utils/config, 修改配置  
```
dataset_path = '/root/lbs/LDB/dataset'  #更换为你本地的数据集位置
train_csv = os.path.join(dataset_path, 'train.csv') 
test_csv = os.path.join(dataset_path, 'test.csv')
train_images = os.path.join(dataset_path, 'train_images')
test_images = os.path.join(dataset_path, 'test_images')
submission_file = '/root/lbs/LDB/submission.csv'  #不用管它

batch_size = 32
epoch_num = 60
learning_rate = 1e-5
devices = 'cuda'
random_seed = 114514
cuda_device = '1,2,3,4,5,6,7'
augment_times = 4
```
打开除后缀是test以外所有以main开头的python文件，将schedule文件更换为
```
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
```

打开utils/dataset.py,取消以下代码的注释（第123行）：  
```
'''if os.path.exists('./dataset/tmp_train_class'):
                shutil.rmtree('./dataset/tmp_train_class')
            organize_files_by_class('./dataset/tmp_train_split.csv', './dataset/train_images', './dataset/tmp_train_class')'''
``` 
运行任何一个main开头的python文件，在其正常训练开始后立即结束进程。  
将之前提到的第123行代码恢复注释。  
运行
```
bash train_stage1.sh
```
每个模型的单卡训练时间可能在3-6小时左右。  
与此同时，打开models/MPSA， 训练MPSA模型。
```
cd models/MPSA
python main.py
```
你可以在models/MPSA/output或output/中找到MPSA的权重文件，其应该为.bin文件，正常用torch.load加载即可。  
在所有模型都训练完毕后，访问logs/，查看每个模型的权重文件，打开multieval.py并在multi_predict函数中参考原先地址，将model1至5所加载的文件地址分别更换为刚刚训练的模型的地址。  
运行multieval.py，.csv文件会带有‘sub_UniRe+MetaFG+Vit+MPSA+Conv_’的前缀，在submission/文件夹中出现。

