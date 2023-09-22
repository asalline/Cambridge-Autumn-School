### Necessary functions needed for "LPD_train.py" to work.

### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -os
###                  -OpenCv


### Importing packages
import os
import cv2 as cv
import numpy as np
import odl
import torch
import torch.nn as nn


### Function that takes all the images from the path directory and crops them.
### Cropping part is hardcoded to match certain type of pictures. One probably
### needs to change the values.
### Inputs: -'path': path to directory where the images are
###         -'amount_of_images': how many images one wants to get from the
###                              given directory
###         -'scale_number': number of how many pixels does the function skip.
###                          Eg. scale_number = 4 -> every 4th pixel is taken
###                          from the original image
### Outputs: -'all_images': list of all images taken from directory
def get_images(path, amount_of_images='all', scale_number=1):

    all_images = []
    all_image_names = os.listdir(path)
    print(len(all_image_names))
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
            
    return all_images


### Function that defines mathematical background for the script.
### More precise function defines geometry for the Radon transform.
### Inputs: -'setup': determines what kind of geometry one wants. Possible
###                   choices are 'full', 'sparse', 'limited'. Default: 'full'
###         -'min_domain_corner': Determines where the bottom left corner is
###                               is in the geometry. Default: [-1,-1]
###         -'max_domain_corner': Determines where the upper right corner is
###                               is in the geometry. Default: [1,1]
###         -'shape': how many points there is in x- and y-axis between the
###                   corners of the geometry. Default: (100,100)
###         -'source_radius': radius of the 'object' when taken measurements.
###                           Default: 2
###         -'detector_radius': radius of the ??? when taken measurements.
###                             Default: 1
###         -'dtype': Python data type. Default: 'float32'
###         -'device': Device which is used in calculations. Default: 'cpu'
###         -'factor_lines': Parameter which controls the line-measurements
###                          Default: 1
### Outputs: -'domain': odl domain, not really used, could be deleted from
###                     the outputs
###          -'geometry': odl geometry, could be deleted from the outputs
###          -'ray_transform': Radon transform operator defined by
###                            given geometry
###          -'output_shape': Shape defined by angles and lines in geometry.
###                           Needed in the allocations.
def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cuda', factor_lines = 1):

    device = 'astra_' + device
    print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (360, int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 3/4
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(512/factor_lines))
        
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape

class LPD_step(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.device = device

        ### Primal block of the network
        self.primal_step = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        ### Dual block of the network
        self.dual_step = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.to(device)

    ### Must needed forward function
    def forward(self, f_rec_images, g_sinograms, h):

        ### Dual iterate happens here
        f_sinogram = self.operator(f_rec_images) / self.operator_norm
        u = torch.cat([h, f_sinogram, g_sinograms / self.operator_norm], dim=1)
        h = h + self.dual_step(u)

        ### Primal iterate happens here
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

        ### Initializing the parameters for every unrolled iteration step.
        for k in range(self.n_iter):
            step = LPD_step(operator, adjoint_operator, operator_norm, device=self.device)
            setattr(self, f'step{k}', step)
            
    def forward(self, f_rec_images, g_sinograms):

        ### Initializing "h" as a zero matrix
        h = torch.zeros(g_sinograms.shape).to(self.device)

        ### Here happens the unrolled iterations
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            f_rec_images, h = step(f_rec_images, g_sinograms, h)
            
        return f_rec_images

