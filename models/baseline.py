import torch.nn as nn
from transformers import AutoModelForImageClassification
import os



class ButterflyCNN(nn.Module):
    def __init__(self, num_classes):
        super(ButterflyCNN, self).__init__()
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        #self.model = models.vit_b_16(pretrained=True)
        
        model_name = "/root/cyh/google/vit-large-patch16-224"

        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.fc = nn.Linear(1000, num_classes)
        self.name = 'baseline'

    def forward(self, x):
        #raise RuntimeError(self.model(x).shape)
        #print(self.model(x).logits.shape)
        return self.fc(self.model(x).logits)