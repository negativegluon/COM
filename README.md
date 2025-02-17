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

VisionTransformer：**等待队友写作**  
  
MPSA：在models/MPSA中，按照MPSA公开仓库中Training部分操作  

### 下载数据集


