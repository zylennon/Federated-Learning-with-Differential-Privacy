# Application of FL task
from MLModel import *
from FLModel import *
from utils import *
import warnings
import time
from tqdm import tqdm
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
def plot_and_save_results(acc, tot_T):
    # 创建一个包含每轮训练准确度的列表
    epochs = list(range(1, tot_T + 1))

    # 绘制准确度图
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, acc, marker='o', linestyle='-')
    plt.title('Accuracy vs. Global Epochs')
    plt.xlabel('Global Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True)
    timestamp = int(time.time())
    output_filename= f"acc_result_{timestamp}.png"
    # 保存图像到文件
    plt.savefig(output_filename)
    print(f"Accuracy plot saved to {output_filename}")
    plt.show()
def load_cnn_mnist(num_users):
    train = datasets.MNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.MNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float().unsqueeze(1)
    test_label = test.targets
    test_data = (test_data - mean) / std

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx]
        targets = train_label[idx].float()
        non_iid.append((d, targets))
    non_iid.append((test_data.float(), test_label.float()))
    return non_iid
def load_cifar(num_users):
    train=datasets.CIFAR10
"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""
client_num = 20
d = load_cnn_mnist(client_num)


"""
FL model parameters.
"""

warnings.filterwarnings("ignore")

lr = 0.15

fl_param = {
    'output_size': 10,
    'client_num': client_num,
    'model': MNIST_CNN,
    'data': d,
    'lr': lr,
    'E': 10,
    'C': 1,
    'eps': 4.0,
    'delta': 1e-5,
    'q': 0.01,
    'clip': 0.1,
    'tot_T': 500,
    'batch_size': 128,
    'device': device
}
print('client_num:'+str(client_num))
fl_entity = FLServer(fl_param).to(device)
acc = []
start_time = time.time()
newacc=0
for t in tqdm(range(fl_param['tot_T'])):
    # newacc = fl_entity.global_update()
    # print("global_update:")
    # print(newacc)
    #
    # if newacc<0.80 :
    #     newacc=fl_entity.global_update()
    # else :
    #     newacc=fl_entity.fedswap_global_update(t,2,2)
    #     print("swap_update:")
    newacc = fl_entity.fedswap_global_update(t, 2, 2)
    acc += [newacc]
    print("global epochs = {:d}, acc = {:.4f}".format(t+1, acc[-1]), " Time taken: %.2fs" % (time.time() - start_time))


plot_and_save_results(acc, fl_param['tot_T'] )