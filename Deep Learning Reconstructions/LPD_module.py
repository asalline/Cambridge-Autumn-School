import torch
import torch.nn as nn


### Function needed when defining the UNet encoding and decoding parts
def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

class LPD_step(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.device = device
        
        
        self.primal_step = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.dual_step = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        # self.to(device)
        
    def forward(self, f_rec_images, g_sinograms, h):
        
        
        f_sinogram = self.operator(f_rec_images) / self.operator_norm
        u = torch.cat([h, f_sinogram, g_sinograms / self.operator_norm], dim=1)
        h = h + self.dual_step(u)
        
        adjoint_eval = self.adjoint_operator(h) / self.operator_norm
        u = torch.cat([f_rec_images, adjoint_eval], dim=1)
        f_rec_images = f_rec_images + self.primal_step(u)
        
        return f_rec_images, h
        
class LPD(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, n_iter, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.n_iter = n_iter
        self.device = device
        
        for k in range(self.n_iter):
            step = LPD_step(operator, adjoint_operator, operator_norm)
            setattr(self, f'step{k}', step)
            
    def forward(self, f_rec_images, g_sinograms):
        
        h = torch.zeros(g_sinograms.shape).to(self.device)
        
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            f_rec_images, h = step(f_rec_images, g_sinograms, h)
            
        return f_rec_images