### Lib


```python
import random
import pandas as pd
import numpy as np
import os
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchsummary

import torchvision
import torchvision.models as models
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import winsound as sd
```

### Set Device


```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)
```

### Set Dataset, Dataloader


```python
data_info = pd.read_csv('../data/train.csv')
data_info.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>path</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_000</td>
      <td>./train/TRAIN_000.mp4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_001</td>
      <td>./train/TRAIN_001.mp4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_002</td>
      <td>./train/TRAIN_002.mp4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform = transforms.Compose([
    transforms.Resize([120, 120]),
    transforms.RandomCrop(112),
    transforms.Grayscale(num_output_channels=3)
    ])
```


```python
class CustomDataset(torch.utils.data.Dataset): 
        def __init__(self, info, transform):
            self.transform = transform
            self.root = '../data/'
            self.pathes = info['path'].values.tolist()
            
            self.y = torch.Tensor(info['label'].values)
            
        def __len__(self):
            return len(self.pathes)
        
        def mapping(self, label):
            if label == 0 : return label+1
            elif label == 1 : return label-1
            elif label == 2 : return label+1
            elif label == 3 : return label-1
            else : return label
        
        def __getitem__(self, idx):
            # n_frame, H, W, channel
            self.x = torchvision.io.read_video(self.root + self.pathes[idx][1:], start_pts=0, end_pts=1, pts_unit='sec')[0].permute(0, 3, 1, 2)
            
            # Data Augmentation using (even frame, odd frame) / p = 0.5
            if random.randint(0, 9) >= 5:
                self.x = self.x[np.arange(0,30,2)]
            else :
                self.x = self.x[np.arange(0,30,2)+1]
                
            self.x = self.transform(self.x)
            trans_y = self.y[idx]
            
            # Data Augmentation using (VerticalFlip-class 0,1, HorigonFlip-class-2,3,4) / p = 0.5
            # if trans_y == 0 or trans_y == 1:
            #     if random.randint(0, 9) >= 5:
            #         self.x = transforms.RandomVerticalFlip(1)(self.x)
            #         trans_y = self.mapping(trans_y)
            
            if trans_y == 2 or trans_y == 3 :
                if random.randint(0, 9) >= 5:
                    self.x = transforms.RandomHorizontalFlip(1)(self.x)
                    trans_y = self.mapping(trans_y)
            
            return self.x, trans_y.to(torch.long)
```


```python
total_dataset = CustomDataset(data_info, transform=transform)

train_dataset, test_dataset = train_test_split(total_dataset, train_size=0.8, stratify = torch.Tensor(data_info['label'].values), random_state = 123)

train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True)
```

### Data Visualization


```python
label_name =['volume_up', 'volume_down',
            'Jump_before', 'Jump_after',
            'stop']
```


```python
for i, (frames, label) in enumerate(train_loader):
    
    rows = 2; cols = 3
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(4, 4))
    # print(np.shape(frames))
    print(label_name[int(label[0])])
    axes[0][0].imshow(frames[0][0].permute(1,2,0))
    axes[0][1].imshow(frames[0][5].permute(1,2,0))
    axes[0][2].imshow(frames[0][10].permute(1,2,0))
    axes[1][0].imshow(frames[0][13].permute(1,2,0))
    # axes[1][1].imshow(frames[0][4].permute(1,2,0))
    # axes[1][2].imshow(frames[0][5].permute(1,2,0))
    break
```

    Jump_after
    


    
![png](output_11_1.png)
    


### Modeling


```python
resnet18_pretrained = models.resnet18(pretrained=True)
```

    C:\Users\user\anaconda3\envs\anomal\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and will be removed in 0.15, please use 'weights' instead.
      warnings.warn(
    C:\Users\user\anaconda3\envs\anomal\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and will be removed in 0.15. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
      warnings.warn(msg)
    


```python
renet_output_layer = 32
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Linear(num_ftrs, renet_output_layer)
```


```python
class ConvLSTM_classification(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0):
        super().__init__()
        
        self.backbone = resnet18_pretrained
        self.conv1 = nn.Conv1d(15, 32, 4, 2, 0)  # (t-kernersize+1)/stride
        self.conv2 = nn.Conv1d(32, 64, 4, 2, 0)  # (t-kernersize+1)/stride

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*6, 5)
        )
        
        self.conv2d = nn.Conv1d(1, 4, 1, 1, 0)
        
    def forward(self, x):
        x_ = [self.backbone(x[:,i]) for i in range(x.size(1))]
        x = torch.stack(x_, dim=1)
        # x : (batch, frame, 32)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.fc(x)
        
        return out
```


```python
resnet18_conv1D = ConvLSTM_classification(32, 64, 2, 0.2).to(DEVICE)
```


```python
# resnet18_conv1D(torch.Tensor(5, 15, 3, 112, 112).to(DEVICE)).size()
```


```python
cost = []
cost_val = []
F1_score_val = []

def train():
    num_epochs = 15
    lr = 1e-4
    model = resnet18_conv1D
    optim = torch.optim.Adam(model.parameters(), lr=lr)# weight_decay = 0.004
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        # model train
        
        label_list = torch.Tensor([])
        pred_list = torch.Tensor([])
        
        model.train()
        Accuracy = 0
        cost_ = 0
        
        for i, (images, labels) in enumerate(train_loader):
            input_ = images.float().to(DEVICE)
            label_ = labels.to(DEVICE)
            
            optim.zero_grad()
    
            out_ = model(input_)
            # loss = criterion(out_.to("cpu"), label_.to("cpu"))
            loss = F.cross_entropy(out_.to("cpu"), label_.to("cpu"))
            
            pred_ = torch.argmax(out_.to("cpu"), dim=1)
            
            label_list = torch.hstack((label_list, label_.to("cpu")))
            pred_list = torch.hstack((pred_list, pred_))
            
            loss.backward()
            optim.step()
            
            cost_ += loss
        
        # Score(cost, acc)
        Accuracy = accuracy_score(label_list, pred_list)
        cost_ = cost_/len(train_loader)
        cost.append(cost_.tolist())
        
        # model evaluation
        
        label_list = torch.Tensor([])
        pred_list = torch.Tensor([])
        
        model.eval()
        with torch.no_grad():
            cost_val_ = 0
            Accuracy_val = 0
            
            for i, (images, labels) in enumerate(test_loader):
                input_ = images.float().to(DEVICE)
                label_ = labels.to(DEVICE)

                out_ = model(input_)
                pred_ = torch.argmax(out_.to("cpu"), dim=1)
                
                loss = F.cross_entropy(out_.to("cpu"), label_.to("cpu"))
                cost_val_ += loss
                
                label_list = torch.hstack((label_list, label_.to("cpu")))
                pred_list = torch.hstack((pred_list, pred_))
                
            # Score(cost, acc, f1)
            Accuracy_val = accuracy_score(label_list, pred_list)
            F1_score = f1_score(label_list, pred_list, average="macro")
            
            cost_val_ = cost_val_/len(test_loader)
            cost_val.append(cost_val_.tolist())
            F1_score_val.append(F1_score.tolist())
            
        if(epoch % 1 == 0 ):
            print("--------------------  Epoch : {:3d}/{:3d}  --------------------".format(epoch, num_epochs))
            print("[train] loss : {:2.4f} Acc : {:2.5f}%".format(cost_, Accuracy))
            print("[test ] loss : {:2.4f} Acc : {:2.5f}% F1 score : {:2.2f}%".format(cost_val_, Accuracy_val, F1_score))
            
        
        
        if(min(cost) == cost_) : torch.save(model, ("./model/resnet18_conv1D_best_train.pt"));
        if(min(cost_val) == cost_val_) : torch.save(model, ("./model/resnet18_conv1D_best_test.pt"));
        if(max(F1_score_val) == F1_score) : torch.save(model, ("./model/resnet18_conv1D_best_f1score.pt"));
```

### Train


```python
train()
sd.Beep(2000, 1000)
```

    --------------------  Epoch :   0/ 15  --------------------
    [train] loss : 1.2150 Acc : 0.52664%
    [test ] loss : 0.4673 Acc : 0.83607% F1 score : 0.84%
    --------------------  Epoch :   1/ 15  --------------------
    [train] loss : 0.4936 Acc : 0.83197%
    [test ] loss : 0.5723 Acc : 0.79508% F1 score : 0.79%
    --------------------  Epoch :   2/ 15  --------------------
    [train] loss : 0.3273 Acc : 0.89344%
    [test ] loss : 0.2336 Acc : 0.90984% F1 score : 0.91%
    --------------------  Epoch :   3/ 15  --------------------
    [train] loss : 0.2844 Acc : 0.90574%
    [test ] loss : 0.1496 Acc : 0.95082% F1 score : 0.95%
    --------------------  Epoch :   4/ 15  --------------------
    [train] loss : 0.1793 Acc : 0.94467%
    [test ] loss : 0.1331 Acc : 0.95902% F1 score : 0.96%
    --------------------  Epoch :   5/ 15  --------------------
    [train] loss : 0.2106 Acc : 0.93238%
    [test ] loss : 0.3445 Acc : 0.83607% F1 score : 0.85%
    --------------------  Epoch :   6/ 15  --------------------
    [train] loss : 0.1364 Acc : 0.95492%
    [test ] loss : 0.1333 Acc : 0.95082% F1 score : 0.95%
    --------------------  Epoch :   7/ 15  --------------------
    [train] loss : 0.1220 Acc : 0.96721%
    [test ] loss : 0.2286 Acc : 0.91803% F1 score : 0.92%
    --------------------  Epoch :   8/ 15  --------------------
    [train] loss : 0.1044 Acc : 0.97131%
    [test ] loss : 0.1583 Acc : 0.94262% F1 score : 0.94%
    --------------------  Epoch :   9/ 15  --------------------
    [train] loss : 0.0877 Acc : 0.98156%
    [test ] loss : 0.2054 Acc : 0.93443% F1 score : 0.93%
    --------------------  Epoch :  10/ 15  --------------------
    [train] loss : 0.0806 Acc : 0.98156%
    [test ] loss : 0.1649 Acc : 0.95082% F1 score : 0.95%
    --------------------  Epoch :  11/ 15  --------------------
    [train] loss : 0.1362 Acc : 0.96721%
    [test ] loss : 0.0811 Acc : 0.96721% F1 score : 0.97%
    --------------------  Epoch :  12/ 15  --------------------
    [train] loss : 0.0586 Acc : 0.97541%
    [test ] loss : 0.1943 Acc : 0.92623% F1 score : 0.93%
    --------------------  Epoch :  13/ 15  --------------------
    [train] loss : 0.0657 Acc : 0.98770%
    [test ] loss : 0.2800 Acc : 0.90164% F1 score : 0.90%
    --------------------  Epoch :  14/ 15  --------------------
    [train] loss : 0.0727 Acc : 0.98156%
    [test ] loss : 0.1618 Acc : 0.92623% F1 score : 0.92%
    


```python
plt.title("train/validation loss")
plt.plot(cost)
plt.plot(cost_val)
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend(['train', 'test'])
```




    <matplotlib.legend.Legend at 0x17f4f1c4be0>




    
![png](output_21_1.png)
    


### Test


```python
resnet18_conv1D = torch.load('./model/resnet18_conv1D_best_train.pt').to(DEVICE)
```


```python
quiz_info = pd.read_csv('../data/test.csv')
quiz_info.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000</td>
      <td>./test/TEST_000.mp4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_001</td>
      <td>./test/TEST_001.mp4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_002</td>
      <td>./test/TEST_002.mp4</td>
    </tr>
  </tbody>
</table>
</div>




```python
transform_quiz = transforms.Compose([
    transforms.Resize([120, 120]),
    transforms.CenterCrop(112),
    transforms.Grayscale(num_output_channels=3)
    ])
```


```python
class QuizDataset(torch.utils.data.Dataset): 
        def __init__(self, info, transform):
            self.transform = transform
            self.root = '../data/'
            self.pathes = info['path'].values.tolist()
            
        def __len__(self):
            return len(self.pathes)

        def img_vectorization(self, imgs):
            # imgs -> torch.size(5,112,112,3)
            for i in range(imgs.size(0)-1):
                imgs[i] = imgs[i] - imgs[i+1]
                
            return imgs[0:imgs.size(0)-1]
        
        def __getitem__(self, idx):
            # n_frame, H, W, channel
            self.x = torchvision.io.read_video(self.root + self.pathes[idx][1:], start_pts=0, end_pts=1, pts_unit='sec')[0].permute(0, 3, 1, 2)[np.arange(0,30,2)+1]
            self.x = self.transform(self.x)
            return self.x
```


```python
quiz_dataset = QuizDataset(quiz_info, transform_quiz)
quiz_loader = DataLoader(quiz_dataset, batch_size = 153, shuffle = False)
```


```python
quiz_answer = torch.Tensor([])

for i, images in enumerate(quiz_loader):
    with torch.no_grad():
        input_ = images.float().to(DEVICE)

        out_ = resnet18_conv1D(input_)
        out_ = torch.argmax(out_, dim=1)
        quiz_answer = torch.hstack([quiz_answer, out_.to('cpu')])
```


```python
quiz_df = pd.DataFrame(quiz_answer, columns=['Answer'])
quiz_df.to_csv('./Answer.csv')
```


```python
preds = quiz_df['Answer'].values
solution  = pd.read_csv('./test_real_solution.csv')['Label'].values
```


```python
f1_score(preds, solution, average="macro")
```




    0.9489268272497849


