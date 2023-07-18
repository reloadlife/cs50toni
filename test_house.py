
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
class PricePredictor(nn.Module):
    def __init__(self, image_model, text_model, numerical_dim):
        super(PricePredictor, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.fc1 = nn.Linear(numerical_dim, 512)
        self.fc2 = nn.Linear(1792, 1)  # reduced the size here
        self.image_fc = nn.Linear(2048, 512)
        self.dropout = nn.Dropout(0.5)  # added a Dropout layer

    def forward(self, image, inputs, numerical):
        if image is not None:
            if isinstance(self.image_model, EfficientNet):
                image_features = self.image_model.extract_features(image)
                image_features = torch.mean(image_features, dim=[2, 3])
            else:
                image_features = self.image_model(image)
                image_features = image_features.view(image_features.size(0), -1)

        image_features = self.dropout(self.image_fc(image_features))

        text_features = self.text_model(**inputs)[0][:, 0, :]  # take CLS token
        numerical = self.dropout(self.fc1(numerical))
        print(f"image_features shape: {image_features.shape}")
        print(f"text_features shape: {text_features.shape}")
        print(f"numerical shape: {numerical.shape}")
        numerical = numerical.squeeze(1)
        x = torch.cat((image_features, text_features, numerical), dim=1)
        x = self.fc2(x)
        return x
def model_factory(image_model_name, text_model, numerical_dim):
    image_model = None
    if image_model_name == 'resnet152':
        image_model = resnet152(pretrained=True)
        image_model = nn.Sequential(*list(resnet152(pretrained=True).children())[:-1],
                                    nn.Flatten(),
                                    nn.Linear(2048, 512))
    elif image_model_name.startswith('efficientnet'):
        image_model = EfficientNet.from_pretrained(image_model_name)
    model = PricePredictor(image_model, text_model, numerical_dim)
    return model
text_model = BertModel.from_pretrained('bert-base-uncased')

model = model_factory('efficientnet-b5', text_model, numerical_dim=4)


device = "cuda"
model_path = "best_model_epoch_18.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


image_path = "15011.jpg"
address = "682 Spring Street"
city = "Oak View, CA"
zip_code = 253
beds = 2
baths = 1
sqft = 862
class HouseDataset(Dataset):
    def __init__(self, root_dir, csv_file, tokenizer, transform=None):
        self.dataframe = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.default_image = transforms.ToTensor()(Image.new('RGB', (224, 224)))

        self.scaler_numerical = StandardScaler()
        self.dataframe.iloc[:, 3:7] = self.scaler_numerical.fit_transform(self.dataframe.iloc[:, 3:7])
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
        text = text1 + " [SEP] " + text2  # combine the two texts with a separator
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

        numerical = self.dataframe.iloc[idx, 3:7].values.astype(float)
        numerical = torch.from_numpy(numerical).float()
        price = self.dataframe.iloc[idx, 7]

        sample = {'image': image,
                  'input_ids': inputs['input_ids'].squeeze(),
                  'attention_mask': inputs['attention_mask'].squeeze(),
                  'numerical': numerical, 'price': price}
        return sample

numerical = np.array([zip_code, beds, baths, sqft])
valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = HouseDataset("apex/train", "data.csv", tokenizer, transform=valid_transform)


def predict_price(model, tokenizer, numerical_scaler, price_scaler, instance, image_path):
    img_id, address, city, area, bedrooms, bathrooms, sqft, _ = instance.split(',')

    numerical = np.array([float(area), float(bedrooms), float(bathrooms), float(sqft)])
    numerical = torch.tensor(numerical_scaler.transform([numerical])).float()

    text = address + " " + city
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")

    img_name = os.path.join(image_path, img_id + ".jpg")
    if os.path.exists(img_name):
        image = Image.open(img_name).convert('RGB')
        image = valid_transform(image)
    else:
        image = torch.zeros((3, 224, 224))

    

    device = next(model.parameters()).device
    image = image.to(device)
    numerical = numerical.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        model.eval()
        pred = model(image.unsqueeze(0), inputs, numerical.unsqueeze(0))

    pred = price_scaler.inverse_transform(pred.cpu().numpy())

    return pred[0][0]

model.load_state_dict(torch.load('best_model_epoch_18.pth', map_location=torch.device('cpu')))
dataframe = pd.read_csv(os.path.join("apex/test", "data.csv"))

scaler_numerical = StandardScaler()
scaler_numerical.fit_transform(dataframe.iloc[:, 3:7])
scaler_price = StandardScaler()
scaler_price.fit_transform(dataframe.iloc[:, 7].values[:, None])

sample_data = """
15001,952 Blaine Avenue,Fillmore CA,118,4,2,1704,549000
15002,17208 Village 17,Camarillo CA,59,2,2,1829,599000
15003,6 Livermore Avenue,Ventura CA,390,3,3,2141,599500
15004,405 Prospect Street,Oak View CA,253,2,1,989,588000
15005,939 Breton Avenue,Simi Valley CA,349,3,2,1693,579500
15006,125 Conifer Circle,Oak Park CA,252,3,2.1,1462,590000
15007,411 Howell Road,Oxnard CA,261,3,1,1120,599000
15008,1263 Erringer Road,Simi Valley CA,349,4,2,1890,569000
15009,172 Mountain View Street,Oak View CA,253,3,2,1248,549000
15010,422 Fulton Street,Camarillo CA,59,4,2,1394,589900
15011,682 Spring Street,Oak View CA,253,2,1,862,549000
15012,169 La Veta Drive,Camarillo CA,59,3,2,1443,589000
15013,6219 Calle Bodega,Camarillo CA,59,4,2,1574,600000
15014,346 S Victoria Avenue,Ventura CA,390,3,2,1333,565000
15015,59 W Calle El Prado,Oak View CA,253,4,2,1584,549000
15016,840 Fine Street,Fillmore CA,118,3,2.1,2539,559000
15017,209 Cahuenga Drive,Oxnard CA,261,1,1,560,599000
15018,200 S F Street,Oxnard CA,261,3,2,1485,599900
15019,944 Coronado Circle,Santa Paula CA,338,3,3,2096,559000
15020,107 Fraser Lane,Ventura CA,390,2,2,1567,570000
15021,517 Yarrow Drive,Simi Valley CA,349,3,2.1,1743,598000
15022,362 Autumn Path Lane,Santa Paula CA,338,4,2.1,2377,562991
15023,0 H Street,Oxnard CA,261,3,2.1,1653,575000
15024,1611 Patricia Avenue,Simi Valley CA,349,3,2,1082,599995
15025,3139 Trinity Drive,Ventura CA,390,3,1,1225,624900
15026,4192 Apricot Road,Simi Valley CA,349,3,2,1661,620000
15027,376 Calistoga Road,Camarillo CA,59,3,3,1790,635000
15028,7846 LILAC Lane,Simi Valley CA,349,2,2,1354,625000
15029,982 La Vuelta Place,Santa Paula CA,338,3,2,2232,609000
15030,2152 Eastridge Trail,Oxnard CA,261,3,2,1642,649900
15031,2783 Machado Street,Simi Valley CA,349,3,2,1303,625000
15032,2240 Crestmont Drive,Ventura CA,390,3,2,1216,599900
15033,30101 Village 30,Camarillo CA,59,2,2,1907,627000
15034,1002 Borden Street,Simi Valley CA,349,4,3,2222,629900
15035,1545 E Avenida De Los Arboles,Thousand Oaks CA,372,5,3,1819,599000
15036,545 Murray Avenue,Camarillo CA,59,3,2,1576,595000
15037,1824 Garvin Avenue,Simi Valley CA,349,4,3,1957,600000
15038,425 Vista Del Sol,Camarillo CA,59,4,2.1,1828,599000
15039,33 Nottingham Road,Westlake Village CA,401,4,2,1848,598900
"""

csvList = "price\n"
for instance in sample_data.split('\n')[1:-1]:
    predicted = predict_price(model, tokenizer, scaler_numerical, scaler_price, instance, "apex/test")
    csvList += f"{predicted}\n"

print("\n\n")
print(csvList)