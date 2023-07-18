
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50 , resnet152
from efficientnet_pytorch import EfficientNet
from transformers import BertTokenizer, BertModel
from PIL import Image
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import torch.nn.functional as F
device = "cuda"
num_epochs = 100
image_size = 448
linear_nn = 100
lr = 1e-5
patience = 10


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class HouseDatasetBert(Dataset):
    def __init__(self, root_dir, csv_file, tokenizer, transform=None):
        self.dataframe = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.default_image = transforms.ToTensor()(Image.new('RGB', (224, 224)))

        self.scaler = StandardScaler()
        self.dataframe.iloc[:, 3:7] = self.scaler.fit_transform(self.dataframe.iloc[:, 3:7])
        self.scaler_price = StandardScaler()
        self.dataframe.iloc[:, 7] = self.scaler_price.fit_transform(self.dataframe.iloc[:, 7].values[:, None])
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.dataframe.iloc[idx, 0]) + ".jpg")

        image = None
        if os.path.exists(img_name):
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
        else:
            image = self.default_image

        text1, text2 = self.dataframe.iloc[idx, 1], self.dataframe.iloc[idx, 2]
        text = text1 + " [SEP] " + text2
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

        numerical = self.dataframe.iloc[idx, 3:7].values.astype(float)
        numerical = torch.from_numpy(numerical).float()
        price = self.dataframe.iloc[idx, 7]

        sample = {'image': image,
                  'input_ids': inputs['input_ids'].squeeze(),
                  'attention_mask': inputs['attention_mask'].squeeze(),
                  'numerical': numerical, 'price': price}
        return sample

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

"""
DFD
"""
class PricePredictorBERT(nn.Module):
    def __init__(self, image_model, text_model, numerical_dim):
        super(PricePredictorBERT, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.fc1 = nn.Linear(numerical_dim, 512)
        self.fc2 = nn.Linear(1792, 1)
        self.image_fc = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image, inputs, numerical):
        if image is not None:
            if isinstance(self.image_model, EfficientNet):
                image_features = self.image_model.extract_features(image)
                image_features = torch.mean(image_features, dim=[2, 3])
            else:
                image_features = self.image_model(image)
                image_features = image_features.view(image_features.size(0), -1)

        image_features = self.dropout(self.image_fc(image_features))

        text_features = self.text_model(**inputs)[0][:, 0, :]
        numerical = self.dropout(self.fc1(numerical))

        x = torch.cat((image_features, text_features, numerical), dim=1)
        x = self.fc2(x)
        return x


class PricePredictorLSTM(nn.Module):
    def __init__(self, image_model, vocab_size, embedding_dim, hidden_dim,numerical_dim):
        super(PricePredictorLSTM, self).__init__()
        self.image_model = image_model
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(numerical_dim, 512)
        self.fc2 = nn.Linear(2048 + hidden_dim + 512, 1)
        self.image_fc = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.4)

    def forward(self, image, inputs, numerical):
        if image is not None:
            if isinstance(self.image_model, EfficientNet):
                image_features = self.image_model.extract_features(image)
                image_features = torch.mean(image_features, dim=[2, 3])
            else:
                image_features = self.image_model(image)
                image_features = image_features.view(image_features.size(0), -1)

        image_features = self.dropout(self.image_fc(image_features))

        input_ids = inputs['input_ids']
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        text_features = lstm_out[:, -1, :]

        numerical = self.dropout(self.fc1(numerical))

        x = torch.cat((image_features, text_features, numerical), dim=1)
        x = self.fc2(x)
        return x
#
"""
SP2
"""

data1 = pd.read_csv('apex/train/data.csv')
data2 = pd.read_csv('apex/test/data.csv')

text_data = pd.concat([data1['street'], data1['citi'], data2['street'], data2['citi']])
tokens = text_data.str.split(' ').explode()

vocab_size = tokens.nunique() + 4

print('Vocabulary size:', vocab_size)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_model = BertModel.from_pretrained('bert-base-uncased')

train_dataset = HouseDatasetBert("apex/train", "data.csv", tokenizer, transform=train_transform)
valid_dataset = HouseDatasetBert("apex/test", "data.csv", tokenizer, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False,drop_last=True)
def model_factory(image_model_name,nlp_model, vocab_size,embedding_dim,hidden_dim, numerical_dim,text_model=None):
    image_model = None
    if image_model_name == 'resnet152':
        image_model = nn.Sequential(*list(resnet152(pretrained=True).children())[:-1],
                                    nn.Flatten(),
                                    nn.Linear(2048, 512))
    elif image_model_name.startswith('efficientnet'):
        image_model = EfficientNet.from_pretrained(image_model_name)
        if nlp_model == "bert":
            model = PricePredictorBERT(image_model, text_model, numerical_dim)
        elif nlp_model == "lstm":
            model = PricePredictorLSTM(image_model,vocab_size , embedding_dim,hidden_dim, numerical_dim)

    return model

embedding_dim = 50
hidden_dim = 100
# model = model_factory('efficientnet-b5',"lstm",vocab_size,embedding_dim,hidden_dim,numerical_dim=4)
model = model_factory('efficientnet-b5', "bert", vocab_size, embedding_dim, hidden_dim, numerical_dim=4,text_model=text_model)
model.to(device)

class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, true):
        return torch.mean(torch.abs((true - pred) / true)) * 100

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5,weight_decay=0.01)
criterion_mse = nn.MSELoss()
criterion_mape = MAPELoss()
scheduler = CosineAnnealingLR(optimizer, T_max=10)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')



def train_epoch(train_loader, model, criterion, optimizer, device, accumulation_steps=2):
    model.train()
    running_loss = 0.0
    total_mape = 0.0
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training")

    for i, batch in enumerate(pbar):
        images = batch['image'].to(device)
        inputs1 = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
        numerical = batch['numerical'].to(device)
        price = batch['price'].to(device)

        outputs = model(images, inputs1, numerical)

        outputs = outputs.float()
        price = price.float()
        loss = criterion(outputs.view(-1), price)
        loss = loss / accumulation_steps
        loss.backward()

        running_loss += loss.item() * images.size(0) * accumulation_steps
        total_mape += mape(price.detach().cpu().numpy(), outputs.view(-1).detach().cpu().numpy()) * images.size(0)

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({'running_loss': running_loss / ((pbar.n + 1) * train_loader.batch_size),
                          'running_mape': total_mape / ((pbar.n + 1) * train_loader.batch_size)})

    if (i+1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    return running_loss / len(train_loader.dataset), total_mape / len(train_loader.dataset)
def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    total_mape = 0.0
    pbar = tqdm(valid_loader, desc="Validation")
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            text = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
            numerical = batch['numerical'].to(device)
            price = batch['price'].to(device)

            outputs = model(images, text, numerical)
            outputs = outputs.float()
            price = price.float()
            loss = criterion(outputs.view(-1), price)

            running_loss += loss.item() * images.size(0)
            total_mape += mape(price.detach().cpu().numpy(), outputs.view(-1).detach().cpu().numpy()) * images.size(0)

            pbar.set_postfix({'running_loss': running_loss / ((pbar.n + 1) * valid_loader.batch_size),
                              'running_mape': total_mape / ((pbar.n + 1) * valid_loader.batch_size)})
    return running_loss / len(valid_loader.dataset), total_mape / len(valid_loader.dataset)

best_loss = float('inf')
n_patience = 0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    train_loss = train_epoch(train_loader, model, criterion_mape, optimizer, device)
    print(train_loss)

    print('Train Loss: {:.4f}'.format(train_loss[0]))
    valid_loss = valid_epoch(valid_loader, model, criterion_mape, device)
    print('Valid Loss: {:.4f}'.format(valid_loss[0]))

    scheduler.step()
    valid_loss = valid_loss[0]
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(best_model_wts, f'best_model_epoch_{epoch+1}.pth')
        n_patience = 0
    else:
        n_patience += 1

    if n_patience > patience:
        print("Early stopping...")
        break

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_model.pth')