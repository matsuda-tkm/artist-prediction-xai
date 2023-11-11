from torch import nn
import torch.nn.functional as F

class CharacterCNN(nn.Module):
    def __init__(self, num_classes ,embed_size=128, max_length=200, filter_sizes=(2, 3, 4, 5), filter_num=64):
        super().__init__()
        self.params = {'num_classes': num_classes ,'embed_size':embed_size, 'max_length':max_length, 'filter_sizes':filter_sizes, 'filter_num':filter_num}
        self.embed_size = embed_size
        self.max_length = max_length
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num

        self.embedding = nn.Embedding(0xffff, embed_size)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_size, filter_num, filter_size) for filter_size in filter_sizes
        ])
        self.fc1 = nn.Linear(filter_num * len(filter_sizes), 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1,2)

        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(embedded))
            pooled = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        convs_merged = torch.cat(conv_outputs, dim=1)
        fc1_output = F.relu(self.fc1(convs_merged))
        bn_output = self.batch_norm(fc1_output)
        do_output = self.dropout(bn_output)
        fc2_output = self.fc2(do_output)
        return fc2_output

class CharacterCNNEmbedding(nn.Module):
    def __init__(self, embed_size=256):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(0xffff, embed_size)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1,2)
        return embedded

class CharacterCNNClassifier(nn.Module):
    def __init__(self, num_classes, embed_size=256, filter_sizes=(2, 3, 4, 5), filter_num=64):
        super().__init__()
        self.embed_size = embed_size
        self.filter_sizes = filter_sizes
        self.filter_num = filter_num

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_size, filter_num, filter_size) for filter_size in filter_sizes
        ])
        self.fc1 = nn.Linear(filter_num * len(filter_sizes), 64)
        self.batch_norm = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = F.relu(conv_layer(x))
            pooled = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        convs_merged = torch.cat(conv_outputs, dim=1)
        fc1_output = F.relu(self.fc1(convs_merged))
        bn_output = self.batch_norm(fc1_output)
        do_output = self.dropout(bn_output)
        fc2_output = self.fc2(do_output)
        return fc2_output