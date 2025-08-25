import os
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from datetime import datetime
import matplotlib.pyplot as plt
import re

class Trainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, criterion,
                 dataloader, template_verts, template_faces, template_edges, device,
                 checkpoint_dir, start_epoch=0):
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        self.template_verts = template_verts
        self.template_faces = template_faces
        self.template_edges = template_edges
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.start_epoch = start_epoch
        self.best_loss = float('inf')
        self.training_history = []
        self.epoch_losses = []
        self.plot_frequency = 10

    def find_latest_checkpoint(self):
        """Find the latest checkpoint file"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, '*.pth'))
        if not checkpoint_files:
            return None, 0
        
        # Sort by creation time (most recent first)
        checkpoint_files.sort(key=os.path.getctime, reverse=True)
        latest_checkpoint = checkpoint_files[0]
        
        # Extract epoch from filename
        epoch_match = re.search(r'epoch_(\d+)', os.path.basename(latest_checkpoint))
        epoch = int(epoch_match.group(1)) if epoch_match else 0
        
        return latest_checkpoint, epoch

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with detailed logging"""
        print(f"🔄 Loading checkpoint: {os.path.basename(checkpoint_path)}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model states
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
            # Load training history
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', [])
            
            epoch = checkpoint.get('epoch', 0)
            
            print(f"✅ Checkpoint loaded successfully!")
            print(f"   📊 Loaded from epoch: {epoch}")
            print(f"   🎯 Best loss so far: {self.best_loss:.6f}")
            print(f"   📈 Training history entries: {len(self.training_history)}")
            
            return epoch
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return 0

    def check_and_load_checkpoint(self):
        """Check for existing checkpoints and load if found"""
        print("🔍 Checking for existing checkpoints...")
        
        checkpoint_path, epoch_from_name = self.find_latest_checkpoint()
        
        if checkpoint_path:
            print(f"✅ Found checkpoint: {os.path.basename(checkpoint_path)}")
            loaded_epoch = self.load_checkpoint(checkpoint_path)
            print(f"🚀 RESUMING training from epoch {loaded_epoch + 1}")
            return loaded_epoch
        else:
            print("📂 No existing checkpoints found")
            print(f"🆕 STARTING FRESH training from epoch 1")
            return 0

    # ... rest of your existing methods (plot_training_losses, save_checkpoint, train_epoch) ...
    
    def plot_training_losses(self):
        """Plot training losses during training"""
        if len(self.epoch_losses) < 2:
            return

        epochs = [e['epoch'] for e in self.epoch_losses]
        g_losses = [e['g_loss'] for e in self.epoch_losses]
        d_losses = [e['d_loss'] for e in self.epoch_losses]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, g_losses, 'b-', label='Generator Loss')
        plt.plot(epochs, d_losses, 'r-', label='Discriminator Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, g_losses, 'b-', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Generator Loss')
        plt.title('Generator Loss Trend')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_checkpoint(self, epoch, is_best=False):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'hiegan_epoch_{epoch}_{timestamp}.pth'
        if is_best:
            filename = f'hiegan_best_{epoch}_{timestamp}.pth'
        filepath = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'best_loss': self.best_loss,
            'training_history': self.training_history
        }

        torch.save(checkpoint, filepath)
        print(f'💾 Saved checkpoint: {filename}')

    def train_epoch(self, epoch):
        # ... your existing train_epoch code ...
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
        batch_count = 0

        for batch_idx, data in enumerate(pbar):
            imgs, gt_meshes, _ = data
            imgs = imgs.to(self.device)
            gt_meshes = gt_meshes.to(self.device)

            batch_size = imgs.shape[0]
            sample_points = torch.randn(batch_size, 1024, 3, device=self.device) * 0.5

            # Train Discriminator
            self.d_optimizer.zero_grad()

            with torch.no_grad():
                outputs = self.generator(imgs, self.template_verts, self.template_edges, sample_points)
                if 'deformed_vertices' in outputs:
                    deformed_vertices = outputs['deformed_vertices']
                    fake_mesh = Meshes(deformed_vertices, self.template_faces.expand(batch_size, -1, -1))
                    fake_points = sample_points_from_meshes(fake_mesh, 1024).to(self.device)
                else:
                    continue

            real_points = sample_points_from_meshes(gt_meshes, 1024).to(self.device)

            d_real = self.discriminator(real_points)
            d_fake = self.discriminator(fake_points)

            d_loss_real = nn.BCEWithLogitsLoss()(d_real, torch.ones_like(d_real))
            d_loss_fake = nn.BCEWithLogitsLoss()(d_fake, torch.zeros_like(d_fake))
            d_loss = (d_loss_real + d_loss_fake) * 0.5

            d_loss.backward()
            self.d_optimizer.step()

            # Train Generator
            self.g_optimizer.zero_grad()
            outputs = self.generator(imgs, self.template_verts, self.template_edges, sample_points)

            if 'deformed_vertices' in outputs:
                deformed_vertices = outputs['deformed_vertices']
                fake_mesh = Meshes(deformed_vertices, self.template_faces.expand(batch_size, -1, -1))
                fake_points = sample_points_from_meshes(fake_mesh, 1024).to(self.device)
            else:
                continue

            d_pred_fake = self.discriminator(fake_points)
            g_adv_loss = nn.BCEWithLogitsLoss()(d_pred_fake, torch.ones_like(d_pred_fake))

            try:
                g_recon_loss, _ = self.criterion(outputs, gt_meshes)
            except Exception:
                g_recon_loss = nn.MSELoss()(fake_points, real_points)

            g_loss = g_recon_loss + 0.1 * g_adv_loss

            g_loss.backward()
            self.g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            batch_count += 1

            pbar.set_postfix({
                'G_loss': total_g_loss / batch_count,
                'D_loss': total_d_loss / batch_count
            })

        avg_g_loss = total_g_loss / batch_count
        avg_d_loss = total_d_loss / batch_count
        return avg_g_loss, avg_d_loss

    def train(self, num_epochs, start_epoch=0):
        for epoch in range(start_epoch, num_epochs):
            print(f"📊 Starting epoch {epoch + 1}/{num_epochs}")
            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            print(f"   Epoch {epoch +1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

            self.epoch_losses.append({
                'epoch': epoch + 1,
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            })

            is_best = False
            if avg_g_loss < self.best_loss:
                self.best_loss = avg_g_loss
                is_best = True
                print(f"   🎯 New best model! Loss: {avg_g_loss:.4f}")

            if (epoch + 1) % 5 == 0 or is_best or (epoch +1) == num_epochs:
                self.save_checkpoint(epoch + 1, is_best)

            if (epoch + 1) % self.plot_frequency == 0:
                print(f"📊 Plotting training progress...")
                self.plot_training_losses()

def train(generator, discriminator, g_optimizer, d_optimizer, criterion,
          dataloader, template_verts, template_faces, template_edges,
          device, checkpoint_dir, start_epoch=0, epochs=10, plot_at_end=True):
    
    print("=" * 60)
    print("HIE-GAN TRAINING STARTED")
    print("=" * 60)
    
    # Create trainer
    trainer = Trainer(
        generator, discriminator, g_optimizer, d_optimizer, criterion,
        dataloader, template_verts, template_faces, template_edges,
        device, checkpoint_dir, start_epoch=start_epoch)
    
    # Check and load existing checkpoint
    actual_start_epoch = trainer.check_and_load_checkpoint()
    
    print(f"📅 Training configuration:")
    print(f"   Total epochs requested: {epochs}")
    print(f"   Starting from epoch: {actual_start_epoch + 1}")
    print(f"   Remaining epochs: {epochs - actual_start_epoch}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    # Start training
    trainer.train(epochs, start_epoch=actual_start_epoch)
    
    if plot_at_end:
        trainer.plot_training_losses()
    
    print("=" * 60)
    print("HIE-GAN TRAINING COMPLETED!")
    print("=" * 60)
