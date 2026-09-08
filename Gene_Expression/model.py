import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Pix2Feature(nn.Module):
    '''
    Using pre-trained CNN model to extract feature
    '''
    def __init__(self, cnn_arch):
        super(Pix2Feature, self).__init__()
        
        self.cnn = timm.create_model(cnn_arch, pretrained = True, num_classes = 0, global_pool = '')

    def forward(self, x):
        out = self.cnn(x)
        return out

# Patch-level prediction
class Pix2Pathway_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('resnet101', pretrained=True, num_classes=0, global_pool='')
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32768, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1387)
        )
    def forward(self, img):
        # img: patches (batch size, 3, 128, 128)
        out = self.cnn(img)
        out = self.fc(out)
        return out

# Still developing...
class Pix2Pathway_Attn(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.cnn_block = timm.create_model('resnet101', pretrained=True, num_classes=0, global_pool='')
        self.attention = nn.Sequential(
            nn.Linear(2048*4*4, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1387)
        )
    
    def forward(self, x):
        '''
        x: a list contains k patches
        '''
        embeddings = []
        for i in range(self.k):
            f = self.cnn_block(x[i])
            f = nn.Flatten()(f)
            embeddings.append(f)
        #embeddings = torch.tensor(embeddings)
        embeddings = torch.cat(embeddings)
        alpha = self.attention(embeddings)
        alpha = torch.transpose(alpha, 1, 0)
        alpha = F.softmax(alpha, dim = 1)
        M = torch.mm(alpha, embeddings)
        out = self.fc(M)
        return out

# Baseline model using faeture as input
# flatten K features and directly connect 2 fc layers
class Feature2Pathway_baseline(nn.Module):
    def __init__(self, k, size):
        super().__init__()

        # Neuron amounts of each layer
        if size == 'large':
            params = {'fc1': 1024, 'fc2': 1024}
        elif size == 'medium':
            params = {'fc1': 512, 'fc2': 128}
        elif size == 'small':
            params = {'fc1': 256, 'fc2': 64}

        self.fc = nn.Sequential(
            nn.Linear(6144*k, params['fc1']),
            nn.BatchNorm1d(params['fc1']),
            nn.ReLU(),
            nn.Linear(params['fc1'], params['fc2']),
            nn.BatchNorm1d(params['fc2']),
            nn.ReLU(),
            nn.Linear(params['fc2'], 1387)
        )

    def forward(self, features):
        # features: (batch size, k, 6144)
        features = features.view(features.size(0), -1)
        return self.fc(features)

# Using faeture as input
# 2 1D-CNN layers -> flatten -> 2 fc layers
class Feature2Pathway_CNN(nn.Module):
    def __init__(self, k, size):
        super().__init__()
        self.cnn_block = nn.Sequential(
            nn.Conv1d(6144, 1024, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Conv1d(1024, 512, kernel_size=1, stride=1, bias=True),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Neuron amounts of each layer
        if size == 'large':
            params = {'fc_1': 1024, 'fc_2': 1024}
        elif size == 'medium':
            params = {'fc_1': 512, 'fc_2': 128}
        elif size == 'small':
            params = {'fc_1': 256, 'fc_2': 64}

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*k, params['fc_1']),
            BatchNorm1d(params['fc_1']),
            nn.ReLU(),
            nn.Linear(params['fc_1'], params['fc_2']),
            nn.BatchNorm1d(params['fc_2']),
            nn.ReLU(),
            nn.Linear(params['fc_2'], 1387)
        )

    def forward(self, x):
        # x: (batch size, k, 6144)
        x = x.permute(0, 2, 1)   # (batch size, 6144, k)   (In Pytorch, channel should be first)
        x = self.cnn_block(x)   # (batch size, 512, k)
        x = x.permute(0, 2, 1)   # (batch size, k, 512)
        x = self.fc(x)
        return x

# Self-attention pooling layer (be used in Feature2Pathway_SAPool_fc)
class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # x: (batch size, k, 6144)
        alpha = self.W(x)
        alpha = F.softmax(alpha, dim=-1)

        out = torch.sum(x * alpha, dim=1)

        return out

# Using faeture as input
# Self-attention pooling -> 2 fc layers
class Feature2Pathway_SAPool_fc(nn.Module):
    def __init__(self, k, size):
        super().__init__()
        self.sa_pool = SelfAttentionPooling(6144)
        
        if size == 'large':
            params = {'fc1': 1024, 'fc2': 1024}
        elif size == 'medium':
            params = {'fc1': 512, 'fc2': 128}
        elif size == 'small':
            params = {'fc1': 256, 'fc2': 64}
        
        self.fc = nn.Sequential(
            nn.Linear(6144, params['fc1']),
            nn.BatchNorm1d(params['fc1']),
            nn.ReLU(),
            nn.Linear(params['fc1'], params['fc2']),
            nn.BatchNorm1d(params['fc2']),
            nn.ReLU(),
            nn.Linear(params['fc2'], 1387)
        )
    
    def forward(self, x):
        x = self.sa_pool(x)
        x = self.fc(x)
        return x
        
# Using faeture as input
class Feature2Pathway_Attn(nn.Module):
    # https://arxiv.org/pdf/1802.04712.pdf
    def __init__(self, k, size, class_num, gated=False):
        super().__init__()
        if size == 'large':
            params = {'attn_dim': 1024, 'fc1': 1024, 'fc2': 1024}
        elif size == 'medium':
            params = {'attn_dim': 1024, 'fc1': 512, 'fc2': 128}
        elif size == 'small':
            params = {'attn_dim': 512, 'fc1': 256, 'fc2': 64}
        
        if gated:
            self.attention_V = nn.Sequential(
                nn.Linear(6144, params['attn_dim']),
                nn.Tanh()
            )
            self.attention_U = nn.Sequential(
                nn.Linear(6144, params['attn_dim']),
                nn.Sigmoid()
            )
            self.attention_weights = nn.Linear(params['attn_dim'], 1)
        else:
            self.attention = nn.Sequential(
                nn.Linear(6144, params['attn_dim']),
                nn.Tanh(),
                nn.Linear(params['attn_dim'], 1)
            )

        self.fc = nn.Sequential(
            nn.Linear(6144, params['fc1']),
            nn.ReLU(),
            nn.Linear(params['fc1'], params['fc2']),
            nn.ReLU(),
            nn.Linear(params['fc2'], class_num)
        )

        self.gated = gated

    def forward(self, x):
        # batch size must be 1 !!!
        x = x.squeeze(0)   # (k, 6144)

        if self.gated:
            alpha_V = self.attention_V(x)   # (k, params['attn_dim'])
            alpha_U = self.attention_U(x)   # (k, params['attn_dim'])
            alpha = self.attention_weights(alpha_V*alpha_U)   # element wise muliplication (k, 1)
            alpha = torch.transpose(alpha, 1, 0)   # (1, k)
            alpha = F.softmax(alpha, dim=1)

            M = torch.mm(alpha, x)   # (1, 6144)

            out = self.fc(M)
        else:
            alpha = self.attention(x)   # (k, 1)
            alpha = torch.transpose(alpha, 1, 0)   # (1, k)
            alpha = F.softmax(alpha, dim=1)

            M = torch.mm(alpha, x)   # (1, 6144)

            out = self.fc(M)   # (1, 1387)
        
        return out, alpha
