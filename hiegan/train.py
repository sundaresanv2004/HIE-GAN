import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from datetime import datetime

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
        print(f'Saved checkpoint: {filename}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', [])
        print(f'Loaded checkpoint from {checkpoint_path}')
        return checkpoint.get('epoch', 0)

    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0
        total_chamfer = 0

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
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch +1} - G Loss: {avg_g_loss:.4f}, D Loss: {avg_d_loss:.4f}")

            is_best = False
            if avg_g_loss < self.best_loss:
                self.best_loss = avg_g_loss
                is_best = True

            if (epoch + 1) % 5 == 0 or is_best or (epoch +1) == num_epochs:
                self.save_checkpoint(epoch + 1, is_best)


def train(
    generator, discriminator, 
    g_optimizer, d_optimizer, 
    criterion, 
    dataloader, 
    template_verts, 
    template_faces, 
    template_edges, 
    device, 
    checkpoint_dir, 
    start_epoch=0, 
    epochs=10
):
    trainer = Trainer(
        generator, discriminator, g_optimizer, d_optimizer, criterion,
        dataloader, template_verts, template_faces, template_edges,
        device, checkpoint_dir, start_epoch=start_epoch)
    trainer.train(epochs)