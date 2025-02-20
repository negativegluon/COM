import argparse
import os
import torch

dataset_path = './dataset'  #蝴蝶数据集
raise RuntimeError('请修改这里的路径！')
train_csv = os.path.join(dataset_path, 'train.csv') 
test_csv = os.path.join(dataset_path, 'test.csv')
train_images = os.path.join(dataset_path, 'train_images')
test_images = os.path.join(dataset_path, 'test_images')
submission_file = '/root/lbs/LDB/submission.csv'  #这个不用改了

batch_size = 32
epoch_num = 70
learning_rate = 1e-5
devices = 'cuda'
random_seed = 114514
cuda_device = '0,1,2,3,4,5,6,7'
augment_times = 4

def config_args():
    
    parser = argparse.ArgumentParser(description="Configuration for training")

    parser.add_argument('--dataset_path', type=str, default=dataset_path, help='Path to the dataset')
    parser.add_argument('--train_csv', type=str, default=train_csv, help='Path to the training CSV file')
    parser.add_argument('--test_csv', type=str, default=test_csv, help='Path to the testing CSV file')
    parser.add_argument('--train_images', type=str, default=train_images, help='Path to the training images')
    parser.add_argument('--test_images', type=str, default=test_images, help='Path to the testing images')
    parser.add_argument('--submission_file', type=str, default=submission_file, help='Path to the submission file')

    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=epoch_num, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=learning_rate, help='Learning rate for training')
    parser.add_argument('--device', type=str, default=devices if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=random_seed, help='Random seed')
    parser.add_argument('--cuda_device', type=str, default=cuda_device, help='Set the device you want')
    parser.add_argument('--augment_times', type=int, default=augment_times, help='augment times')
    
    args = parser.parse_args()
    return args
