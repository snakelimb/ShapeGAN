import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from tqdm import tqdm
import glob

# Import your shape generation code
from shape_clean_small import SHAPE_NAMES


# Set random seed for reproducibility
torch.manual_seed(999)
np.random.seed(999)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 512
image_size = 64
nc = 3  # Number of channels (RGB)
n_shapes = 5  # Number of shape types
n_params = 8  # Number of shape parameters
learning_rate_g = 0.0002  # Generator learning rate
learning_rate_d = 0.0001  # Discriminator learning rate
beta1 = 0.5
num_epochs = 400
sample_interval = 5  # Generate samples every 5 epochs
checkpoint_interval = 10  # Save models every 10 epochs
d_updates = 1  # Update discriminator once per generator update
label_smoothing = 0.2  # Label smoothing for real labels

class ShapeDataset(data.Dataset):
    """Dataset for loading existing generated shapes with parameters and shape types"""
    
    def __init__(self, data_root="shape_data", transform=None):
        self.data_root = data_root
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load all data from existing dataset
        print("Loading shape dataset from disk...")
        self.data = []
        
        for shape_id, shape_name in enumerate(SHAPE_NAMES):
            shape_folder = os.path.join(data_root, shape_name)
            image_dir = os.path.join(shape_folder, "images")
            param_file = os.path.join(shape_folder, "parameters.pt")
            
            if not os.path.exists(image_dir) or not os.path.exists(param_file):
                print(f"Warning: Missing data for {shape_name} in {shape_folder}")
                continue
            
            # Load parameters
            param_tensors = torch.load(param_file, map_location='cpu')
            
            # Get image files
            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            
            # Ensure we have matching numbers of images and parameters
            min_count = min(len(image_files), len(param_tensors))
            if len(image_files) != len(param_tensors):
                print(f"Warning: Mismatch in {shape_name} - {len(image_files)} images, {len(param_tensors)} parameters. Using {min_count} samples.")
            
            print(f"Loading {min_count} {shape_name} samples...")
            
            for i in tqdm(range(min_count), desc=f"Loading {shape_name}"):
                # Load image
                img_path = image_files[i]
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                
                # Get parameters
                param_vec = param_tensors[i]
                
                # Create one-hot encoding for shape type
                shape_onehot = torch.zeros(n_shapes)
                shape_onehot[shape_id] = 1.0
                
                self.data.append({
                    'image': img_tensor,
                    'parameters': param_vec,
                    'shape_type': shape_onehot,
                    'shape_id': shape_id
                })
        
        print(f"Loaded {len(self.data)} total samples from {data_root}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def weights_init(m):
    """Initialize network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DeterministicGenerator(nn.Module):
    """Deterministic Generator that takes only parameters + shape type (no noise)"""
    
    def __init__(self):
        super(DeterministicGenerator, self).__init__()
        
        # Input size: n_params (parameters) + n_shapes (shape type)
        input_size = n_params + n_shapes
        
        # Projection layer to get to proper size for transposed convolutions
        self.projection = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024 * 4 * 4),
            nn.BatchNorm1d(1024 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.main = nn.Sequential(
            # Input: 1024 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State: 512 x 8 x 8
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 16 x 16
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 32 x 32
            
            nn.ConvTranspose2d(128, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output: nc x 64 x 64
        )
    
    def forward(self, parameters, shape_type):
        # Concatenate parameters and shape type (no noise)
        x = torch.cat([parameters, shape_type], dim=1)
        
        # Project to proper size
        x = self.projection(x)
        x = x.view(x.size(0), 1024, 4, 4)
        
        # Generate image
        output = self.main(x)
        return output

class Discriminator(nn.Module):
    """Conditional Discriminator that takes image + parameters + shape type"""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # Image processing branch
        self.image_conv = nn.Sequential(
            # Input: nc x 64 x 64
            nn.Conv2d(nc, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 64 x 32 x 32
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 128 x 16 x 16
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 256 x 8 x 8
            
            nn.Conv2d(256, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # State: 1024 x 4 x 4
        )
        
        # Condition processing
        self.condition_fc = nn.Sequential(
            nn.Linear(n_params + n_shapes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4 * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, parameters, shape_type):
        # Process image
        img_features = self.image_conv(image)
        img_features = img_features.view(img_features.size(0), -1)
        
        # Process conditions
        condition = torch.cat([parameters, shape_type], dim=1)
        cond_features = self.condition_fc(condition)
        
        # Combine image and condition features
        combined = torch.cat([img_features, cond_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        return output.view(-1, 1).squeeze(1)

def create_sample_grid(generator, device, epoch, fixed_params, fixed_shapes):
    """Generate a grid of sample images"""
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_params, fixed_shapes)
        
        # Create grid WITHOUT automatic normalization to avoid double normalization
        grid = vutils.make_grid(fake_images, nrow=8, normalize=False, scale_each=False)
        
        # Convert to numpy and save
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        grid_np = np.clip(grid_np, 0, 1)  # Ensure values are in valid range
        
        plt.figure(figsize=(12, 12))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title(f'Generated Shapes - Epoch {epoch}')
        
        # Save the plot
        os.makedirs('dcgan_samples', exist_ok=True)
        plt.savefig(f'dcgan_samples/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    generator.train()

def save_checkpoint(netG, netD, optimizerG, optimizerD, epoch, G_losses, D_losses):
    """Save model checkpoints with training state"""
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': netG.state_dict(),
        'discriminator_state_dict': netD.state_dict(),
        'optimizer_G_state_dict': optimizerG.state_dict(),
        'optimizer_D_state_dict': optimizerD.state_dict(),
        'G_losses': G_losses,
        'D_losses': D_losses
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def train_deterministic_gan(data_root="shape_data"):
    """Main training function for deterministic GAN"""
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = ShapeDataset(data_root=data_root)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create networks
    print("Initializing networks...")
    netG = DeterministicGenerator().to(device)
    netD = Discriminator().to(device)
    
    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)
    
    print(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")
    
    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate_g, betas=(beta1, 0.999))
    
    # Fixed parameters and shape types for sample generation
    fixed_params = []
    fixed_shapes = []
    samples_per_shape = 64 // n_shapes
    
    # Sample from each shape type in the dataset
    shape_samples = {i: [] for i in range(n_shapes)}
    for sample in dataset.data:
        shape_id = sample['shape_id']
        if len(shape_samples[shape_id]) < samples_per_shape:
            shape_samples[shape_id].append(sample)
    
    for shape_id in range(n_shapes):
        for sample in shape_samples[shape_id]:
            fixed_params.append(sample['parameters'])
            fixed_shapes.append(sample['shape_type'])
    
    # Fill remaining samples if needed
    while len(fixed_params) < 64:
        sample = dataset.data[0]
        fixed_params.append(sample['parameters'])
        fixed_shapes.append(sample['shape_type'])
    
    fixed_params = torch.stack(fixed_params).to(device)
    fixed_shapes = torch.stack(fixed_shapes).to(device)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    G_losses = []
    D_losses = []
    
    real_label = 1.0 - label_smoothing  # Label smoothing for real labels
    fake_label = 0.0
    
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            real_images = batch['image'].to(device)
            real_params = batch['parameters'].to(device)
            real_shapes = batch['shape_type'].to(device)
            
            batch_size_curr = real_images.size(0)
            
            # Add noise to real images for discriminator regularization
            noise_factor = 0.1 * max(0, 1 - epoch / 50)  # Decrease noise over time
            real_images_noisy = real_images + noise_factor * torch.randn_like(real_images)
            real_images_noisy = torch.clamp(real_images_noisy, -1, 1)
            
            ############################
            # Update Discriminator
            ############################
            for _ in range(d_updates):
                netD.zero_grad()
                
                # Train with real data
                label_real = torch.full((batch_size_curr,), real_label, dtype=torch.float, device=device)
                # Add label noise
                label_real += 0.05 * torch.randn_like(label_real)
                label_real = torch.clamp(label_real, 0, 1)
                
                output_real = netD(real_images_noisy, real_params, real_shapes)
                errD_real = criterion(output_real, label_real)
                errD_real.backward()
                
                # Train with fake data (deterministic generation)
                fake_images = netG(real_params, real_shapes)
                
                # Add noise to fake images too
                fake_images_noisy = fake_images + noise_factor * torch.randn_like(fake_images)
                fake_images_noisy = torch.clamp(fake_images_noisy, -1, 1)
                
                label_fake = torch.full((batch_size_curr,), fake_label, dtype=torch.float, device=device)
                # Add label noise
                label_fake += 0.05 * torch.randn_like(label_fake)
                label_fake = torch.clamp(label_fake, 0, 1)
                
                output_fake = netD(fake_images_noisy.detach(), real_params, real_shapes)
                errD_fake = criterion(output_fake, label_fake)
                errD_fake.backward()
                
                errD = errD_real + errD_fake
                optimizerD.step()
            
            ############################
            # Update Generator
            ############################
            netG.zero_grad()
            
            # Generate fake images for generator update (deterministic)
            fake_images = netG(real_params, real_shapes)
            
            label_gen = torch.full((batch_size_curr,), 1.0, dtype=torch.float, device=device)  # No smoothing for generator
            output_gen = netD(fake_images, real_params, real_shapes)
            errG = criterion(output_gen, label_gen)
            errG.backward()
            optimizerG.step()
            
            # Statistics
            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()
            num_batches += 1
            
            # Print discriminator outputs for debugging
            if i == 0 and epoch < 5:
                with torch.no_grad():
                    print(f"D(real): {output_real.mean().item():.4f}, D(fake): {output_fake.mean().item():.4f}")
        
        # Average losses for the epoch
        avg_d_loss = epoch_d_loss / (num_batches * d_updates)
        avg_g_loss = epoch_g_loss / num_batches
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}')
        
        # Generate sample images every sample_interval epochs
        if (epoch + 1) % sample_interval == 0:
            create_sample_grid(netG, device, epoch + 1, fixed_params, fixed_shapes)
            print(f"Sample images saved for epoch {epoch + 1}")
        
        # Save checkpoint every checkpoint_interval epochs
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(netG, netD, optimizerG, optimizerD, epoch + 1, G_losses, D_losses)
    
    # Save final models
    print("Saving trained models...")
    torch.save(netG.state_dict(), 'deterministic_generator_final.pth')
    torch.save(netD.state_dict(), 'deterministic_discriminator_final.pth')
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig('deterministic_training_losses.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Training completed!")
    return netG, netD

def generate_custom_samples(generator_path='deterministic_generator_final.pth', data_root="shape_data", num_samples=16):
    """Generate samples with custom parameters from existing dataset"""
    
    # Load the trained generator
    netG = DeterministicGenerator().to(device)
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netG.eval()
    
    # Load some parameters from the existing dataset
    dataset = ShapeDataset(data_root=data_root)
    
    custom_params = []
    custom_shapes = []
    
    samples_per_shape = num_samples // n_shapes
    
    # Sample parameters from each shape type
    shape_samples = {i: [] for i in range(n_shapes)}
    for sample in dataset.data:
        shape_id = sample['shape_id']
        if len(shape_samples[shape_id]) < samples_per_shape:
            shape_samples[shape_id].append(sample)
    
    for shape_id in range(n_shapes):
        for sample in shape_samples[shape_id]:
            custom_params.append(sample['parameters'])
            custom_shapes.append(sample['shape_type'])
    
    custom_params = torch.stack(custom_params).to(device)
    custom_shapes = torch.stack(custom_shapes).to(device)
    
    # Generate samples deterministically
    with torch.no_grad():
        generated_images = netG(custom_params, custom_shapes)
        
        # Create and save grid
        grid = vutils.make_grid(generated_images, nrow=samples_per_shape, normalize=False, scale_each=False)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np + 1) / 2
        grid_np = np.clip(grid_np, 0, 1)  # Ensure values are in valid range
        
        plt.figure(figsize=(15, 10))
        plt.imshow(grid_np)
        plt.axis('off')
        plt.title('Deterministic Generated Shapes by Type')
        plt.savefig('deterministic_generated_shapes.png', dpi=150, bbox_inches='tight')
        plt.show()

def test_determinism(generator_path='deterministic_generator_final.pth', data_root="shape_data"):
    """Test that the generator produces identical outputs for identical inputs"""
    
    # Load the trained generator
    netG = DeterministicGenerator().to(device)
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netG.eval()
    
    # Load a single sample
    dataset = ShapeDataset(data_root=data_root)
    sample = dataset.data[0]
    
    params = sample['parameters'].unsqueeze(0).to(device)
    shape_type = sample['shape_type'].unsqueeze(0).to(device)
    
    # Generate the same image multiple times
    print("Testing determinism...")
    images = []
    with torch.no_grad():
        for i in range(5):
            img = netG(params, shape_type)
            images.append(img)
    
    # Check if all images are identical
    all_same = True
    for i in range(1, len(images)):
        if not torch.allclose(images[0], images[i], atol=1e-6):
            all_same = False
            break
    
    if all_same:
        print("✅ Generator is deterministic - identical inputs produce identical outputs")
    else:
        print("❌ Generator is not deterministic - identical inputs produce different outputs")
    
    return all_same

def load_checkpoint(checkpoint_path, netG, netD, optimizerG, optimizerD):
    """Load a checkpoint to resume training"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    netG.load_state_dict(checkpoint['generator_state_dict'])
    netD.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizer_D_state_dict'])
    
    epoch = checkpoint['epoch']
    G_losses = checkpoint['G_losses']
    D_losses = checkpoint['D_losses']
    
    print(f"Checkpoint loaded: Epoch {epoch}")
    return epoch, G_losses, D_losses

if __name__ == "__main__":
    print("Deterministic Conditional GAN for Shape Generation")
    print("=" * 50)
    
    # Train the model
    trained_g, trained_d = train_deterministic_gan(data_root="shape_data")
    
    # Test determinism
    #print("\nTesting determinism...")
    #test_determinism(data_root="shape_data")
    
    # Generate custom samples
    print("\nGenerating custom samples...")
    generate_custom_samples(data_root="shape_data")
    
    print("\nTraining and generation complete!")