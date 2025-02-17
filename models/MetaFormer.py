import os
import torch
import importlib
import torch.distributed as dist


def relative_bias_interpolate(checkpoint):
    for k in list(checkpoint['model']):
        if 'relative_position_index' in k:
            del checkpoint['model'][k]
        if 'relative_position_bias_table' in k:
            relative_position_bias_table = checkpoint['model'][k]
            cls_bias = relative_position_bias_table[:1,:]
            relative_position_bias_table = relative_position_bias_table[1:,:]
            size = int(relative_position_bias_table.shape[0]**0.5)
            img_size = (size+1)//2
            if 'stage_3' in k:
                downsample_ratio = 16
            elif 'stage_4' in k:
                downsample_ratio = 32
            new_img_size = 224//downsample_ratio
            new_size = 2*new_img_size-1
            if new_size == size:
                continue
            relative_position_bias_table = relative_position_bias_table.reshape(size,size,-1)
            relative_position_bias_table = relative_position_bias_table.unsqueeze(0).permute(0,3,1,2)#bs,nhead,h,w
            relative_position_bias_table = torch.nn.functional.interpolate(
                relative_position_bias_table, size=(new_size, new_size), mode='bicubic', align_corners=False)
            relative_position_bias_table = relative_position_bias_table.permute(0,2,3,1)
            relative_position_bias_table = relative_position_bias_table.squeeze(0).reshape(new_size*new_size,-1)
            relative_position_bias_table = torch.cat((cls_bias,relative_position_bias_table),dim=0)
            checkpoint['model'][k] = relative_position_bias_table
    return checkpoint


def load_pretained(model,logger=None,strict=False):
    
    checkpoint = torch.load('/root/lbs/LDB/models/metafg_2_21k_224.pth', map_location='cpu')
    if 'model' not in checkpoint:
        if 'state_dict_ema' in checkpoint:
            checkpoint['model'] = checkpoint['state_dict_ema']
        else:
            checkpoint['model'] = checkpoint
    if True:
        if 'head.weight' in checkpoint['model'] and 'head.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"==============> drop head....................")
            del checkpoint['model']['head.weight']
            del checkpoint['model']['head.bias']
        if 'head.fc.weight' in checkpoint['model'] and 'head.fc.bias' in checkpoint['model']:
            if logger is not None:
                logger.info(f"==============> drop head....................")
            del checkpoint['model']['head.fc.weight']
            del checkpoint['model']['head.fc.bias']
    if True:
        if logger is not None:
            logger.info(f"==============> drop meta head....................")
        for k in list(checkpoint['model']):
            if 'meta' in k:
                del checkpoint['model'][k]
            
    checkpoint = relative_bias_interpolate(checkpoint)
    if 'point_coord' in checkpoint['model']:
        if logger is not None:
            logger.info(f"==============> drop point coord....................")
        del checkpoint['model']['point_coord']
    msg = model.load_state_dict(checkpoint['model'], strict=strict)
    del checkpoint
    torch.cuda.empty_cache()