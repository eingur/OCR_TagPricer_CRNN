from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torch

class CRNN(nn.Module):
    def __init__(self, num_classes=10, hidden_size=128, num_layers=2, pretrained = False, freeze_grads = False):
        super(CRNN, self).__init__()
        self._is_pretrained = pretrained
        if self._is_pretrained:
            resnet = models.resnet18(weights = 'ResNet18_Weights.DEFAULT')
        else:
            resnet = models.resnet18()
            
        resnet_modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*resnet_modules)
        
        if freeze_grads:
            for params in self.resnet.parameters(): #градиенты фризим
                params.requires_grad = False

        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(8,3), stride=2, padding=1), #такое ядро, чтобы красивая чиселка в линейный слой зашла 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )        

        self.linear1 = nn.Linear(1024, hidden_size)
        
        # LSTM для обработки последовательности
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)

        # Слой для классификации (11 классов: цифры от 0 до 9 и спец символ)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        # print(f'{x.shape} after resnet') #([batch_size, 512, 8, 8])

        x = self.conv(x)
        # print(f'{x.shape} after conv')
        batch_size, channels, height,width = x.shape
        
        x = x.view(batch_size,width,  channels * height)  
        #print(x.shape) # ([16,7,1024])
        x = self.linear1(x)
        
        # Проходим через LSTM
        x, _ = self.lstm(x)

        # Прогоняем через классификатор
        x = self.classifier(x)
        
        return x