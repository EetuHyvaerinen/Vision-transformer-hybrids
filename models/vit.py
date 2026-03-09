import torch
import torch.nn as nn

def vit_init_weights(m): 
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, EmbedLayer):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)
    elif isinstance(m, EmbedLayer_distilled):
        nn.init.trunc_normal_(m.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(m.pos_embedding, mean=0.0, std=0.02)

class EmbedLayer(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose(1, 2)
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x

class EmbedLayer_distilled(nn.Module):
    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2 + 1 + 1, embed_dim), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape([x.shape[0], x.shape[1], -1])
        x = x.transpose(1, 2)
        x = torch.cat((
            torch.repeat_interleave(self.cls_token, x.shape[0], 0), 
            torch.repeat_interleave(self.dist_token, x.shape[0], 0), 
            x
        ), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, n_attention_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_attention_heads = n_attention_heads
        self.head_embed_dim = embed_dim // n_attention_heads
        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads)
        self.out_projection = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        b, s, e = x.shape
        xq = self.queries(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        xk = self.keys(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        xv = self.values(x).reshape(b, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        xk = xk.transpose(-1, -2)
        x_attention = torch.matmul(xq, xk)
        x_attention /= float(self.head_embed_dim) ** 0.5
        x_attention = torch.softmax(x_attention, dim=-1)
        x = torch.matmul(x_attention, xv)
        x = x.transpose(1, 2).reshape(b, s, e)
        x = self.out_projection(x)
        return x

class Encoder(nn.Module):
    def __init__(self, embed_dim, n_attention_heads, forward_mul, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim, n_attention_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim * forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * forward_mul, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout1(self.attention(self.norm1(x)))
        x = x + self.dropout2(self.fc2(self.activation(self.fc1(self.norm2(x)))))
        return x

class Classifier(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class Classifier_distilled(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(embed_dim, n_classes)
        self.fc_dist = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        cls_token = x[:, 0, :]
        dist_token = x[:, 1, :]
        x = self.fc1(cls_token)
        x = self.activation(x)
        x = self.fc2(x)
        dist_pred = self.fc_dist(dist_token)
        return (x + dist_pred) / 2

class VisionTransformer(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier(embed_dim, n_classes)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

class MyDeiT(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer_distilled(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = Classifier_distilled(embed_dim, n_classes)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x

class ViT_feature_extractor(nn.Module):
    def __init__(self, n_channels, embed_dim, n_layers, n_attention_heads, forward_mul, image_size, patch_size, n_classes, dropout=0.1):
        super().__init__()
        self.embedding = EmbedLayer(n_channels, embed_dim, image_size, patch_size, dropout=dropout)
        self.encoder = nn.ModuleList([Encoder(embed_dim, n_attention_heads, forward_mul, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(vit_init_weights)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        return x