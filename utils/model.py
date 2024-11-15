import torch
from torch import nn
import torchvision.models as models
from torchinfo import summary
from utils.resnet import *
import torch.nn.functional as F


def init_model(cfg, requires_grad=True):
    """Initialize a model."""
    if cfg.dataset == "linreg":
        model = LinRegModel(cfg)  # 线性回归模型
    elif cfg.model == "mclr":
        model = Mclr_Logistic()
    elif cfg.model == "cnn":
        model = CNN(cfg)  # 卷积神经网络模型
    elif cfg.model == "resnet":
        model = resnet20()  # resnet20 for cifar10 # 使用 ResNet20，用于 CIFAR-10 数据集
        replace_bn_with_ln(model)  # 将模型中的 BN（批量归一化）替换为 LN（层归一化）
        model.loss = get_loss_func(cfg)  # 获取损失函数
    elif cfg.model == "cnn_cifar":
        model = CNNCifar()
    elif cfg.model == "cnn_mnist2":
        model = CNNMnist2()
    else:
        raise ValueError(f"Invalid model.")  # 如果模型类型无效，则抛出异常
    cfg.device = torch.device(cfg.device)  # 设置设备（GPU 或 CPU）
    model.requires_grad_(requires_grad)  # requires gradients or not  # 设置模型参数是否需要梯度
    return model.to(cfg.device)  # 将模型移动到指定设备上


class LinRegModel(nn.Module):
    """Linear regression model."""

    def __init__(self, cfg):
        super().__init__()
        self.c_i = cfg.c_i
        self.linear = nn.Linear(cfg.linreg_dim_data, 1, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return x

    def loss(self, pred, labels):
        """Loss with l2 regularization."""
        loss_fn = nn.MSELoss()
        loss = 0.5 * loss_fn(pred, labels)
        l2_reg = 0
        for param in self.parameters():
            l2_reg += param.square().sum()
        loss += 0.5 * self.c_i * l2_reg
        return loss


class CNN(nn.Module):
    """CNN used in the FedAvg paper."""

    def __init__(self, cfg):
        super().__init__()
        dim_in_channels = {"mnist": 1, "cifar10": 3, "cifar100": 3}
        dim_in_fc = {"mnist": 1024, "cifar10": 1600, "cifar100": 1600}
        self.loss = get_loss_func(cfg)
        self.conv = nn.Sequential(
            nn.Conv2d(dim_in_channels[cfg.dataset], 32, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(dim_in_fc[cfg.dataset], 512),
            nn.ReLU(True),
            nn.Linear(512, cfg.num_classes),  # output layer
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten the data from dim=1
        x = self.fc(x)
        return x


class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Mclr_Logistic, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.loss = nn.NLLLoss()

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class CNNCifar(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMnist2(nn.Module):
    def __init__(self, num_channels=1, num_classes=10, batch_norm=False):
        super(CNNMnist2, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        if batch_norm:
            self.conv2_norm = nn.BatchNorm2d(20)
        else:
            self.conv2_norm = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_norm(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_activation(cfg):
    """Selects the activation function based on the cfg."""
    if cfg.activation == "relu":  # 如果配置中的激活函数为 ReLU
        return nn.ReLU(inplace=True)  # to save memory # 使用 ReLU 激活函数，并设置 inplace=True 以节省内存
    elif cfg.activation == "sigmoid":  # 如果配置中的激活函数为 Sigmoid
        return nn.Sigmoid()  # 使用 Sigmoid 激活函数
    else:
        raise ValueError("Invalid activation function.")


def get_loss_func(cfg):
    """Selects the loss function based on the cfg."""
    if cfg.loss == "mse":
        return nn.MSELoss()
    elif cfg.loss == "cn":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid loss function.")


def replace_bn_with_ln(module):
    """
    Replace bn in resnet with ln, see
    https://arxiv.org/abs/2308.09565
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            setattr(module, name, nn.GroupNorm(num_groups=1, num_channels=num_features))
        else:
            replace_bn_with_ln(child)
