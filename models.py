from torch import flatten, Tensor
from torch.nn import Module, Sequential, Linear, Conv2d, MaxPool2d, ReLU, LocalResponseNorm, Dropout
from torch.nn.init import normal_, ones_, zeros_, xavier_normal_, calculate_gain

class AlexNet(Module):
    def __init__(self, num_classes:int, dropout:float = 0.5) -> None:
        super().__init__()
        self.conv1 = Sequential(
            Conv2d(3, 96, 11, 4),
            ReLU(),
            LocalResponseNorm(5, 1e-4, .75, 2),
            MaxPool2d(2, 2)
        )
        self.conv2 = Sequential(
            Conv2d(96, 256, 5),
            ReLU(),
            LocalResponseNorm(5, 1e-4, .75, 2),
            MaxPool2d(2, 2)
        )
        self.conv3 = Sequential(
            Conv2d(256, 384, 3),
            ReLU()
        )
        self.conv4 = Sequential(
            Conv2d(384, 384, 3),
            ReLU()
        )
        self.conv5 = Sequential(
            Conv2d(384, 256, 3),
            ReLU(),
            MaxPool2d(2, 2)
        )
        self.fc6 = Sequential(
            Linear(1024, 4096),
            ReLU(),
            Dropout(dropout)
        )
        self.fc7 = Sequential(
            Linear(4096, 4096),
            ReLU(),
            Dropout(dropout)
        )
        self.output = Linear(4096, num_classes)

        self.weight_init()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        out = flatten(out, 1, -1)
        out = self.fc7(self.fc6(out))
        return self.output(out) 

    def weight_init(self) -> None:
        for module in self.modules():
            if isinstance(module, (Conv2d, Linear)):
                xavier_normal_(module.weight, calculate_gain("relu"))
                #normal_(module.weight, mean = 0, std = 0.1)
                if module.bias is not None:
                    zeros_(module.bias)
        
        #zeros_(self.conv1[0].bias)
        #zeros_(self.conv3[0].bias)
        
        #ones_(self.conv2[0].bias)
        #ones_(self.conv4[0].bias)
        #ones_(self.conv5[0].bias)
        #ones_(self.fc6[0].bias)
        #ones_(self.fc7[0].bias)
        #ones_(self.output.bias)