import torch
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models


class PAM_Module(nn.Module):
    def __init__(self, in_channels):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.functional.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    def __init__(self, in_channels):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = x.view(batch_size, C, -1)
        proj_key = x.view(batch_size, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.functional.softmax(energy, dim=-1)

        proj_value = x.view(batch_size, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out


class ResNetPAMCAM(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNetPAMCAM, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.pam = PAM_Module(2048)
        self.cam = CAM_Module(2048)
        self.num_classes = num_classes

        # Modify the last fully connected layer
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)

        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)

        out = self.pam(out)
        out = self.cam(out)

        out = self.resnet.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.resnet.fc(out)

        return out
    
def predict_class(image):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = ResNetPAMCAM(num_classes=21).to(device)

    model.load_state_dict(torch.load('weights/model.pt', map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    input_image = Image.open(image).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)

    output_tensor = model(input_tensor)
    output_array = output_tensor.detach().numpy()
    predicted_class_index = output_array.argmax()

    class_labels = ['BAS',
                    'NGS',
                    'MMZ',
                    'MON',
                    'HAC',
                    'NGB',
                    'ART',
                    'PLM',
                    'NIF',
                    'FGC',
                    'MYB',
                    'ABE',
                    'EBO',
                    'OTH',
                    'KSC',
                    'LYT',
                    'BLA',
                    'EOS',
                    'PMO',
                    'LYI',
                    'PEB']
    
    dct = {
        'ABE': 'Abnormal eosinophil',
        'ART': 'Artefact',
        'BAS': 'Basophil',
        'BLA': 'Blast',
        'EBO': 'Erythroblast',
        'EOS': 'Eosinophil',
        'FGC': 'Faggott cell',
        'HAC': 'Hairy cell',
        'KSC': 'Smudge cell',
        'LYI': 'Immature lymphocyte',
        'LYT': 'Lymphocyte',
        'MMZ': 'Metamyelocyte',
        'MON': 'Monocyte',
        'MYB': 'Myelocyte',
        'NGB': 'Band neutrophil',
        'NGS': 'Segmented neutrophil',
        'NIF': 'Not identifiable',
        'OTH': 'Other cell',
        'PEB': 'Proerythroblast',
        'PLM': 'Plasma cell',
        'PMO': 'Promyelocyte'
    }
    
    predicted_class_label = dct[class_labels[predicted_class_index]]

    return predicted_class_label