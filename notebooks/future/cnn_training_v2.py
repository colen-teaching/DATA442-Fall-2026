import os
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import v2

# Check available hardware
if torch.cuda.is_available():
    print(f'GPU is available, using GPU {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
elif torch.mps.is_available():
    print(f'Apple Metal Performance Shaders framework is available, using {torch.device("mps")}')
    device = torch.device('mps')
else:
    print(f'Hardware acceleration not available, using CPU')
    device = torch.device('cpu')

class UTKFaceDataset(Dataset):
    """ Custom dataset class to load in UTK Face dataset elements
        Files have the name format [age]_[gender]_[race]_[date&time].jpg
    """
    def __init__(self, root=os.path.expanduser('~/data10/utk-face/UTKface_inthewild'), transform=None):
        self.root = root
        self.image_files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image = Image.open(os.path.join(self.root, image_file))
        if image.mode != 'RGB': 
            image = image.convert('RGB')
            
        # File name contains the target
        age = int(image_file.split('_')[0])

        if self.transform:
            image, age = self.transform(image, age)

        return image, torch.tensor(age, dtype=torch.float32)

class ResBlock_v2(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()        
        self.conv1 = nn.Conv2d(c_in, c_in // 2, kernel_size=1, padding='same')
        self.bn1 = nn.BatchNorm2d(c_in // 2)

        self.conv2 = nn.Conv2d(c_in // 2, c_in // 2, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(c_in // 2)

        self.conv3 = nn.Conv2d(c_in // 2, c_out, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm2d(c_out)

        if c_in == c_out:
            self.proj = nn.Identity()
        else:
            # Projection layer to match channel dimensions
            self.proj = nn.Conv2d(c_in, c_out, kernel_size=1) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + self.proj(x) # Skip connection
        out = F.relu(out)

        return out

class CNN_Regression_v2(nn.Module):
    def __init__(self):
        super().__init__()

        # Let's do a list of stages, each containing some CNN layers
        # After each stage, we'll do a MaxPool and downsample
        self.stages = nn.ModuleList()

        # Input data is [B, 3, 128, 128]
        # First stage projects data onto a large feature map
        self.stages.append(
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding='same'),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
        )
        # First stage output is [B, 32, 64, 64]
        
        # Let's just stack a bunch of Residual blocks in each stage
        in_channels = 32
        for ii in range(4):
            stage = []
            stage.append(ResBlock_v2(in_channels, 2*in_channels))
            in_channels *= 2
            stage.append(ResBlock_v2(in_channels, in_channels))
            stage.append(ResBlock_v2(in_channels, in_channels))
            self.stages.append(nn.Sequential(*stage))
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        # x has shape [B, 3, H, W]
        # Iterate through my nn.ModuleList of stages
        for stage in self.stages:
            x = stage(x)
            x = F.max_pool2d(x, 2)

        # Spatial average pooling
        x = x.mean(dim=(-2, -1))
        
        x = self.dropout(x)
        x = self.fc(x)
        x = F.relu(x) # Apply relu to ensure non-negative age predictions
        return x   

def train(model, 
          optimizer, 
          loss_func, 
          train_dataset, 
          val_dataset, 
          batch_size=32, 
          num_epochs=10, 
          num_workers=4, 
          scheduler=None,
          early_stop=10):
    model = model.to(device)

    # Build training and validation loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    train_loss, val_loss = [], []
    train_mad, val_mad = [], [] # Tracking mean absolute deviation in units of years

    for epoch in range(num_epochs):
        model.train()
        train_loss_ = 0.
        train_mad_ = 0.
        for xb, yb in tqdm(train_loader, leave=False):
            # Send data to device
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)

            # Make a prediction, compute the loss
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)

            # Backpropagate, apply gradients, reset for next pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log loss and metrics
            train_loss_ += loss.item() * yb.shape[0]
            train_mad_ += F.l1_loss(y_pred, yb, reduction='sum').item()
            
        train_loss.append(train_loss_ / len(train_loader.dataset))
        train_mad.append(train_mad_ / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            val_loss_ = 0.
            val_mad_ = 0.
            for xb, yb in tqdm(val_loader, leave=False):
                # Send data to device
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)

                # Make a prediction, compute the loss
                y_pred = model(xb)
                loss = loss_func(y_pred, yb)

                # Log loss and metrics
                val_loss_ += loss.item() * yb.shape[0]
                val_mad_ += F.l1_loss(y_pred, yb, reduction='sum').item()

            val_loss.append(val_loss_ / len(val_loader.dataset))
            val_mad.append(val_mad_ / len(val_loader.dataset))

        print(f'Epoch {epoch+1} | '
              f'Train loss { train_loss[-1]:.3f} | '
              f'Train MAD { train_mad[-1]:.3f} years | '
              f'Val loss { val_loss[-1]:.3f} | '
              f'Val MAD { val_mad[-1]:.3f} years '
        )

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()

        # Early stopping
        best_epoch = np.argmin(val_loss)
        if epoch - best_epoch > early_stop:
            print(f'Early stopping at Epoch {epoch+1}')

    return train_loss, train_mad, val_loss, val_mad

if __name__ == '__main__':
    # Create the dataset
    # ImageNet Normalization
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    norm = v2.Normalize(mean=imagenet_mean, std=imagenet_std)
    
    denormalize = lambda img: img * imagenet_std[:, None, None] + imagenet_mean[:, None, None]
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([128, 128]),
        v2.ToDtype(torch.float32, scale=True),
        norm
    ])
    dataset = UTKFaceDataset(transform=transform)

    # For reproducibiltiy
    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[0.6, 0.2, 0.2])

    # Create the model
    model = CNN_Regression_v2().to(device)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'CNN has {num_trainable_params} trainable parameters')
    
    # Run the training
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train_loss, train_mad, val_loss, val_mad = train(
        model, optimizer, loss, train_dataset, val_dataset,
        batch_size=64, num_epochs=25, num_workers=8)

    # Save the results
    torch.save({
        'model_weights': model.state_dict(),
        'train_loss': train_loss,
        'train_mad': train_mad,
        'val_loss': val_loss,
        'val_mad': val_mad,
        'test_indices': test_dataset.indices,
    }, os.path.expanduser('~/scr10/cnn_regression_v2_results.pt'))