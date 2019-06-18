import torchvision as tv
import torchvision.transforms as transforms  #数据变换模块
import torch as t
import torch.nn as nn
import torch.nn.functional as Fuc
from torch.autograd import Variable
import time
from torch import optim
from torchvision.transforms import ToPILImage
import torch
import matplotlib.pyplot as plt
import cv2


cuda_available = torch.cuda.is_available()  # 检测GPU是否可用

show = ToPILImage()       #把Tensor转成Image，方便可视化
#数据加载与预处理：归一化为[-1,1]的Tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
#训练集 root为datasets的路径 download为True检测数据集是否已下载，未下载则下载
traindata = tv.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
#数据加载器 batch_size为每次进入数据的多少 shuffle为true则打乱数据顺序
trainloader = t.utils.data.DataLoader(traindata, batch_size=64, shuffle=True, num_workers=0)
#测试集
testdata = tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
testloader = t.utils.data.DataLoader(testdata, batch_size=64, shuffle=True, num_workers=0)
#数据类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#定义卷积神经网络
class CNN(nn.Module):
    def __init__(self):  #python类的继承
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3)  #第一层卷积 卷积核  移动步长为1
        self.conv2 = nn.Conv2d(32, 64, 7, 1, 3)  #第二层卷积 同上
        self.conv3 = nn.Conv2d(64, 88, 7, 1, 3)
        # self.conv4 = nn.Conv2d(24, 48, 3, 1, 1)
        self.fc1 = nn.Linear(88*4*4, 64)  #四个全连接
        self.fc2 = nn.Linear(64, 56)
        self.fc3 = nn.Linear(56, 32)
        self.fc4 = nn.Linear(32, 10)  #最后输出为十个类别

    # 定义前向传播
    def forward(self, x):
        # out = self.conv1(x)
        out = Fuc.max_pool2d(Fuc.relu(self.conv1(x)), 2)  #第一层采用最大值池化
        out = Fuc.avg_pool2d(Fuc.relu(self.conv2(out)), 2) #第二层采用均值池化
        #out = self.conv2(out)
        out = Fuc.max_pool2d(Fuc.relu(self.conv3(out)), 2) #第三层采用最大值池化
        out = out.view(-1, 88*4*4)  #tensor的一个方法，改变size但元素总数不变，将tensor尺寸转化成一维数据
        out = Fuc.relu(self.fc1(out))  #激活函数ReLu：对于输入图像中的每一个负值，PeLU激活函数都返回0值；而对于输入图像中的正值，返回这个正值
        out = Fuc.relu(self.fc2(out))
        out = Fuc.relu(self.fc3(out))
        out = self.fc4(out)
        return out


class VGG_Net(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self, num_classes=10):
        super(VGG_Net, self).__init__()
        self.features = nn.Sequential(
            # 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            # 14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),   #dropout是指在网络的训练过程中，按照一定的概率将网络中的神经元丢弃，这样有效防止过拟合。
            # 15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),   #dropout是指在网络的训练过程中，按照一定的概率将网络中的神经元丢弃，这样有效防止过拟合。
            # 16
            nn.Linear(4096, num_classes),
        )
        # self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        #        print(out.shape)
        out = out.view(out.size(0), -1)
        #        print(out.shape)
        out = self.classifier(out)
        #        print(out.shape)
        return out


def printCon(con):
    plt.figure()
    for i,f in enumerate(con[0].weight.data):
        plt.subplot(12,10,i+1)
        plt.imshow(f[0,:,:],camp='gray')
        plt.axis('off')

net=CNN()
#net=VGG_Net()
if cuda_available == True:
    net.cuda()
print(net)  #打印构建好的网络结构

#定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  #交叉熵损失函数
# optimizer_1 = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  #带动量的随机梯度下降
optimizer_2 = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)   #momentum：当前一次效果好的话就加快步伐；当前一次效果不好就减缓步伐。还可以跳出局部最优值
#训练网络
s_time = time.time()
z = 1   ##设置次数
loss1 = []
for epoch in range(z):  #规定训练循环次数
    # if epoch <= int(z/2):
        # optimizer = optimizer_1
    # elif epoch > int(z/2):
    optimizer = optimizer_2
    r_loss = 0.0
    print("一轮循环计算中……")
    for i, data in enumerate(trainloader, 0):  #得到索引和数据
        #输入数据
        inputs, labels = data  #得到数据和标签并赋值
        if cuda_available == True:
            inputs, labels = inputs.cuda(), labels.cuda()
        else:
            inputs, labels = inputs, labels

        #梯度清零 避免反向传播累加上一次梯度
        optimizer.zero_grad()

        #经过卷积神经网络的输出
        outputs = net(inputs)

        #计算损失值
        loss = criterion(outputs, labels)

        #loss反向传播
        loss.backward()

        #优化器更新参数
        optimizer.step()
        r_loss += loss.item()
        if i % 100 == 99:
            print('第%d轮  损失值:%.3f' % (epoch + 1, r_loss / 100))  #平均损失值
            loss1.append(r_loss/100)
            r_loss = 0.0  #清零

    print('正在测试集上测试训练出的网络......')
    correct = 0
    # 总共的图片数
    total = 0
    for data in testloader:
        images, labels = data
        if cuda_available == True:
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(Variable(images))
        _, predicted = t.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('测试集上的准确率为: %d %%' % (100 * correct / total))

    print('正在绘制曲线，请稍后......')
    plt.plot(loss1,color='#70ad47')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #   plt.savefig('loss1.png')
    plt.show()

print('训练完成！')
e_time = time.time()
print("训练花费时间为:", e_time - s_time)

#printCon(net.conv1(inputs))

torch.save(net.state_dict(), './VGG16.pth')

