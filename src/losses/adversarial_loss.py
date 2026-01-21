import torch
import torch.nn as nn

class AdversarialLoss(nn.Module):
    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        Adversarial Loss Module.
        Args:
            type (str): 'lsgan' (Least Squares GAN) or 'vanilla' (BCE).
            target_real_label (float): Label for real data.
            target_fake_label (float): Label for fake data.
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'lsgan':
            self.loss = nn.MSELoss()
        elif type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError(f"Adversarial loss type '{type}' not implemented.")

    def get_target_tensor(self, prediction, target_is_real):
        target_tensor = self.real_label if target_is_real else self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

class GeneratorLoss(nn.Module):
    def __init__(self, adv_loss):
        super(GeneratorLoss, self).__init__()
        self.adv_loss = adv_loss

    def forward(self, fake_pred):
        # Generator wants fake data to be classified as real
        return self.adv_loss(fake_pred, True)

class DiscriminatorLoss(nn.Module):
    def __init__(self, adv_loss):
        super(DiscriminatorLoss, self).__init__()
        self.adv_loss = adv_loss

    def forward(self, real_pred, fake_pred):
        real_loss = self.adv_loss(real_pred, True)
        fake_loss = self.adv_loss(fake_pred, False)
        return (real_loss + fake_loss) * 0.5
