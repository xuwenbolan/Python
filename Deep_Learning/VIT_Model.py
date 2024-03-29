import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import pygame

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

data_train = pd.read_csv('/home/xuwenbo/data/train.csv')
train_data = data_train.drop('label', axis=1).values
train_mean = train_data.mean()/255.
train_std = train_data.std()/255.
eval_count = 1000
train_transform = transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]),
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean], std=[train_std]),
])

def get_image():
    image = Image.open('/home/xuwenbo/data/7_1.jpg')
    return np.array(image)


def draw_image():
    pygame.init()
    canvas_size = (28, 28)
    window_size = (canvas_size[0] * 40, canvas_size[1] * 40)
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    white = (255, 255, 255)
    Black = (0, 0, 0)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption('Drawing Canvas')

    drawing = False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    x, y = event.pos
                    if 0 <= x < window_size[0] and 0 <= y < window_size[1]:
                        canvas[y // 40, x // 40] = 255
        screen.fill(Black)
        for y in range(canvas_size[1]):
            for x in range(canvas_size[0]):
                if canvas[y, x] == 255:
                    pygame.draw.rect(screen, white, (x * 40, y * 40, 40, 40))

        pygame.display.flip()
    saved_array = canvas
    pygame.quit()
    return saved_array

# image = get_image()
image = draw_image()

model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128)
model = model.to(DEVICE)

model_path = '/home/xuwenbo/data/simple_model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
# print(image_array)
tram_image = transform(image)
test_dataset = []
test_dataset.append(tram_image)
submission_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
title = ""
for images in submission_loader:
    images = images.to(DEVICE)
    output = F.log_softmax(model(images), dim=1)
    _, pred = torch.max(output, dim=1)

    for prediction in pred:
        title = "Pre number: " + str(prediction.item())
plt.imshow(image,cmap='gray',interpolation='nearest')
plt.title(title)
plt.show()