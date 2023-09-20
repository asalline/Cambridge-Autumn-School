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

### Class for the Learned Gradient Descent (LGD) algorithm.
class LGS(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, \
                  g_sinograms, f_rec_images, in_channels, out_channels, step_length, n_iter):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.n_iter = n_iter
        
        self.operator = operator_module
        self.gradient_of_f = adjoint_operator_module

        
        LGS_layers = [
            nn.Conv2d(in_channels=self.in_channels, \
                                    out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
                                    kernel_size=(3,3), padding=1),
        ]
        
        self.layers = nn.Sequential(*LGS_layers)
        
        self.layers2 = [self.layers for i in range(n_iter)]
 

    def forward(self, f_rec_images, g_sinograms):
        
        for i in range(self.n_iter):
        
            f_sinogram = self.operator(f_rec_images)
            
            grad_f = self.gradient_of_f(f_sinogram - g_sinograms) # (output of dual - g_sinograms)
            
            u = torch.cat([f_rec_images, grad_f], dim=1)
            
            u = self.layers2[i](u)

            df = -self.step_length * u[:,0:1,:,:]
            
            f_rec_images = f_rec_images + df
        
        return f_rec_images, self.step_length
