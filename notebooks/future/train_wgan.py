import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
from torchvision.transforms import v2
from torchvision.datasets import MNIST

import numpy as np
import os
from tqdm import tqdm

if torch.cuda.is_available():
    print(f'GPU is available, using GPU {torch.cuda.get_device_name(0)}')
    device = torch.device('cuda')
elif torch.mps.is_available():
    print(f'Apple Metal Performance Shaders framework is available, using {torch.device("mps")}')
    device = torch.device('mps')
else:
    print(f'Hardware acceleration not available, using CPU')
    device = torch.device('cpu')

def sample_noise(shape, mode='uniform'):
    if mode == 'uniform':
        return 2 * torch.rand(*shape) - 1
    else:
        return torch.randn(*shape)

class Generator(nn.Module):
    def __init__(self, latent_size=10, hidden_size=128, output_size=28, batch_norm=True):
        super().__init__()

        # Going from [B, L, 1, 1] to [B, 1, 28, 28]
        self.generator = nn.Sequential(
            # [B, L, 1, 1]
            nn.ConvTranspose2d(latent_size, hidden_size, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # [B, H, 4, 4]
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # [B, H, 7, 7]
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            # [B, H, 14, 14]
            nn.ConvTranspose2d(hidden_size, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.output_size = output_size
        self.latent_size = latent_size

    def forward(self, z):
        z = z.reshape([z.shape[0], self.latent_size, 1, 1])
        x = self.generator(z)
        return x

class Discriminator(nn.Module):
    """ We consider the WGAN Critic/Discriminator as the best 
        possible K-Lipschitz function that maps images to real numbers
        1. "Best possible" means we train the discriminator to optimality
        2. "K-Lipschitz" means that the distance between two discriminator
            outputs is bounded by K x the distance between the inputs
    """
    def __init__(self, input_size=28, hidden_size=128, batch_norm=True):
        super().__init__()
        # Go from [B, 1, 28, 28] to #[B, 1, 1, 1]
        self.discriminator = nn.Sequential(
            # [B, 1, 28, 28],
            nn.Conv2d(1, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # [B, H, 14, 14],
            nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # [B, H, 7, 7],
            nn.Conv2d(hidden_size, hidden_size, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            # [B, H, 3, 3],
            nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        y = self.discriminator(x)
        return y.reshape([y.shape[0], 1])


def train_discriminator(x_real, generator, discriminator, optimizer, lipschitz_method='clip', grad_lambda=0.01):
    b = x_real.shape[0]
    discriminator.zero_grad()

    # Evaluate discriminator on true samples
    y_real = discriminator(x_real)
    loss_real = -y_real.mean()

    # Generate fake samples
    z_fake = sample_noise([b, generator.latent_size]).to(x_real.device)
    x_fake = generator(z_fake)

    # Evaluate discriminator on fake samples
    y_fake = discriminator(x_fake)

    # Total loss
    loss = -y_real.mean() + y_fake.mean()

    if lipschitz_method == 'clip':
        # Enforce K-Lipschitz constraint with gradient clipping
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_lambda)
        
    elif lipschitz_method == 'gradient_penalty':
        # Enforce K-Lipschitz constraint using gradient penalty
        
        # Interpolate between real and fake samples
        alpha = torch.rand(b, 1, 1, 1, dtype=torch.float, device=x_real.device, requires_grad=True)
        x_mid = alpha * x_real + (1 - alpha) * x_fake
    
        # Evaluate discriminator on these samples
        y_mid = discriminator(x_mid)
    
        # Get gradient of discriminator w.r.t. inputs
        gradient = torch.autograd.grad(
            inputs=x_mid,
            outputs=y_mid,
            grad_outputs=torch.ones_like(y_mid),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient = gradient.reshape([b, -1])
    
        # Penalize the norm being unequal to 1
        grad_norm = gradient.norm(2, dim=1)
        grad_penalty = torch.mean((grad_norm - 1.) ** 2)
    
        loss = loss + grad_lambda * grad_penalty
        
    loss.backward()
    optimizer.step()

    return loss.item(), y_real.detach().cpu().numpy(), y_fake.detach().cpu().numpy()

def train_generator(x_real, generator, discriminator, optimizer):
    b = x_real.shape[0]
    generator.zero_grad()

    # Generate fake samples
    z_fake = sample_noise([b, generator.latent_size]).to(x_real.device)
    x_fake = generator(z_fake)

    # Evaluate discriminator on fake samples
    y_fake = discriminator(x_fake)

    loss = -y_fake.mean() # Fool the discriminator
    loss.backward()
    optimizer.step()
    
    return loss.item()

def train_wgan(generator, 
               discriminator,
               opt_generator,
               opt_discriminator,
               dataset, 
               n_critic=5,
               lipschitz_method='clip',
               grad_lambda=0.1,
               batch_size=32, 
               num_epochs=100, 
               num_workers=0, 
               device=device,
               save_epochs=10,
               save_name='WGAN',
               save_path='~/scr10/GAN'):
    # Send model to correct hardware device
    generator = generator.to(device)    
    discriminator = discriminator.to(device)    

    # Create dataloaders for train/validation
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    logging = {
        'dis_loss': [],
        'gen_loss': [],
        'D(real)': [],
        'D(fake)': []
    }

    os.makedirs(os.path.expanduser(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        # Prepare logging
        for key in logging:
            logging[key].append(0)
            
        for batch in tqdm(loader, leave=False):
            # Send x, y to GPU/TPU/etc.
            x = batch[0].float().to(device)

            # Train discriminator to "optimality"
            for _ in range(n_critic):
                dis_loss, y_real, y_fake = train_discriminator(
                    x, 
                    generator, 
                    discriminator, 
                    opt_discriminator, 
                    lipschitz_method=lipschitz_method, 
                    grad_lambda=grad_lambda
                )
                
            # Train the generator
            gen_loss = train_generator(x, generator, discriminator, opt_generator)

            # Logging
            logging['dis_loss'][-1] += dis_loss * x.shape[0]
            logging['gen_loss'][-1] += gen_loss * x.shape[0]
            logging['D(real)'][-1] += y_real.sum()
            logging['D(fake)'][-1] += y_fake.sum()

        outstr = f'Epoch {epoch+1} | '
        for key in logging.keys():
            logging[key][-1] = logging[key][-1] / len(dataset)
            outstr += f'{key} {logging[key][-1]:.3f} | '

        print(outstr)

        if epoch % save_epochs == 0:
            torch.save({
                'generator_weights': generator.state_dict(),
                'discriminator_weights': discriminator.state_dict(),
                **logging,
                
            }, os.path.join(os.path.expanduser(save_path), f'MNIST_{save_name}_epoch={epoch:03d}.pt'))

    return logging

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['WGAN', 'WGAN-GP'], default='WGAN')
    args = parser.parse_args()

    torch.manual_seed(42)
    
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5,), std=(0.5,))
    ])
    mnist_train = MNIST(root='~/scr10/', train=True, transform=transform, download=True)
    print('Training dataset has length', len(mnist_train))

    torch.manual_seed(42)
    if args.model == 'WGAN':
        generator = Generator(batch_norm=True)
        discriminator = Discriminator(batch_norm=True)
        
        opt_generator = torch.optim.RMSprop(generator.parameters(), lr=5e-5)
        opt_discriminator = torch.optim.RMSprop(discriminator.parameters(), lr=5e-5)

        lipschitz_method = 'clip'
        grad_lambda = 0.01
    else:
        generator = Generator(batch_norm=False)
        discriminator = Discriminator(batch_norm=False)
        
        opt_generator = torch.optim.Adam(generator.parameters(), lr=2e-4)
        opt_discriminator = torch.optim.Adam(discriminator.parameters(), lr=2e-4)

        lipschitz_method = 'gradient_penalty'
        grad_lambda = 10.
    
    logging = train_wgan(
        generator, discriminator,
        opt_generator, opt_discriminator,
        mnist_train, 
        device=device, 
        n_critic=5,
        lipschitz_method=lipschitz_method,
        grad_lambda=grad_lambda,
        batch_size=128, 
        num_epochs=100, 
        num_workers=4,
        save_name=args.model,
    )