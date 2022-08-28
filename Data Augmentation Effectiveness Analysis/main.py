#from apex import amp
import copy
import mobilenet_large
import os
import numpy as np
import torch
import time
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random
import vgg
import efficientnet
import resnet
def default_loader(path):
    img = Image.open(path).convert('RGB')
    return img
import time
import torch
import torch.nn as nn
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__()  # 对继承自父类的属性进行初始化
        fh = open(txt, 'r')  # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        imgs = []
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            words = line.split()  # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            imgs.append(
                (words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader

    # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img1, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img1 = self.loader(img1)  # 按照路径读取图片
        if self.transform is not None:
            img1 = self.transform(img1)  # 数据标签转换为Tensor
            return img1,label  # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************
    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 更新混淆矩阵
def confusion_matrix(labels, preds, conf_matix):
    stacked = torch.stack((labels, preds), dim=1)
    for p in stacked:
        tl, pl = p.tolist()
        conf_matix[tl, pl] = conf_matix[tl, pl] + 1
    return conf_matix


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # 设置随机数种子
    setup_seed(3407)
    ckpt_path = r'D:\samm\result\vgg\\'
    if os.path.exists(ckpt_path) is False:
        os.mkdir(ckpt_path)
    BATCH_SIZE = 16
    EPOCH = 100
    train_transforms = transforms.Compose([
        transforms.Resize((256,256),Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((256,256),Image.BICUBIC),
        transforms.ToTensor()
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_txt =r"D:\samm\txt\train\\"
    test_txt = r"D:\samm\txt\test\\"
    max =100
    min =-1
    paths = os.listdir(test_txt)
    paths.sort()
    start_time = time.time()
    for i,path in enumerate(paths):
        #print(i,path.split('.')[0])
        if i>min and i<max :
            print(i,path)
            train_dataset = MyDataset(txt=train_txt + path[:-4]+"_train.txt", transform=train_transforms)
            test_dataset = MyDataset(txt=test_txt + path, transform=test_transforms)
            train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=True)
            test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,num_workers=4, shuffle=False)
            #model = resnet.resnet50().to(device)
            model = vgg.vgg().to(device)
            #model = efficientnet.effnetv2_s().to(device)
            loss_func = torch.nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=0.0001)
            #model, optimizer = amp.initialize(model, opt, opt_level="O1")
            loss = 0
            loss_count = []
            train_acc = 0
            test_acc = 0
            TrainAcc_count = []
            TestAcc_count = []
            print(TestAcc_count)
            best_loss = 0
            best_test_acc = 0
            best_train_acc = 0
            for epoch in range(1, EPOCH + 1):
                print(time.time())
                train_correct = 0
                test_correct = 0
                all_targets = torch.tensor([]).to(device)
                all_preds = torch.tensor([]).to(device)
                model.train()
                time0 = time.time()
                for j, (x, y) in enumerate(train_dataloader):
                    x,y = x.to(device),y.to(device)
                    train_out = model(x)
                    loss = loss_func(train_out, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_predicted = torch.max(train_out.data, 1)[1]
                    train_correct += (train_predicted.cpu().numpy() == y.cpu().numpy()).sum()
                train_acc = train_correct / len(train_dataset)
                TrainAcc_count.append(train_acc)
                loss_count.append(loss)
                with torch.no_grad():
                    model.eval()
                    for a, b in test_dataloader:
                        a, b = a.to(device), b.float().to(device)
                        all_targets = torch.cat((all_targets.float(), b.view(-1, 1)), dim=0)
                        test_out = model(a)
                        test_predicted = torch.max(test_out.data, 1)[1]
                        test_predicted = test_predicted.float()
                        test_correct += (test_predicted.cpu().numpy() == b.cpu().numpy()).sum()
                        all_preds = torch.cat((all_preds, test_predicted.view(-1, 1)), dim=0)
                test_acc = test_correct / len(test_dataset)
                TestAcc_count.append(test_acc)
                if test_acc >= best_test_acc:
                    best_loss = loss
                    best_train_acc = train_acc
                    best_test_acc = test_acc
                    best_epoch = epoch
                    best_targets = all_targets
                    best_preds = all_preds
                    best_wts = copy.deepcopy(model.state_dict())
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| train accuracy: %.2f' % train_acc,
                      '| test accuracy: %.2f' % test_acc,'| estimate time: %d h' % (int((time.time()-time0))*(max-i-1)*EPOCH//3600))
                if test_acc==1.00:
                    break
            print('Best Epoch: ', best_epoch, '| Best train average accuracy: %.2f' % best_train_acc,
                  '| Best test average accuracy: %.2f' % best_test_acc)
            model.load_state_dict(best_wts)
            torch.save(model.state_dict(), ckpt_path  +path[:-4]  + '.pt')
            #plt.figure('Accuracy')
            #plt.plot(TrainAcc_count, label='Train_Accuracy')
            #plt.plot(TestAcc_count, label='Test_Accuracy')
            #plt.savefig(ckpt_path + path.split('.')[0]+'_'+str(best_test_acc)+'_VGG_Accuracy.jpg')
            #plt.clf()
