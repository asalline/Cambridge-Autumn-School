### This is a training script for the LGS network.

### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -matplotlib
###                  -LGS_train_module.py (NEEDS ITS OWN PACKAGES EG. OpenCV)
###

### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGS_train_module import get_images, geometry_and_ray_trafo, LGD
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Using function "get_images" to import images from the path.
images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images=100, scale_number=2)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)

### Using functions from "LGS_train_module". Taking shape from images to produce
### odl parameters and getting Ray transform operator and its adjoint.
shape = (np.shape(images)[1], np.shape(images)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)

### Defining FBP operator
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

partial_x = odl.PartialDerivative(domain, axis=0)
partial_y = odl.PartialDerivative(domain, axis=1)
regularizer = OperatorModule(partial_x.adjoint * partial_x + partial_y.adjoint * partial_y).to(device)


### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

### Making sinograms from the images using Radon transform module
sinograms = ray_transform_module(images)

### Allocating used tensors
noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
rec_images = torch.zeros((sinograms.shape[0], ) + shape)

### Defining variables which define the amount of training and testing data
### being used. The training_scale is between 0 and 1 and controls how much
### training data is taken from whole data
training_scale = 1
amount_of_data = sinograms.shape[0]
n_train = int(np.floor(training_scale * amount_of_data))
n_test = int(np.floor(amount_of_data - n_train))

mean = 0
percentage = 0.05

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(np.shape(sinograms)[0]):
    sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinogram = sinogram_k + noise
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

### All the data into same device
sinograms = sinograms[:,None,:,:].to(device)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)

f_images = images[0:n_train]
g_sinograms = noisy_sinograms[0:n_train]
f_rec_images = rec_images[0:n_train]

test_images = get_images('/scratch2/antti/summer2023/test_walnut', amount_of_images='all', scale_number=2)

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

list_of_test_images = list(range(0,363,5))

test_sinograms = ray_transform_module(test_images)

test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
test_rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)

for k in range(np.shape(test_sinograms)[0]):
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.std(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                                                  

test_rec_images = fbp_operator_module(test_noisy_sinograms)
    
test_sinograms = test_sinograms[:,None,:,:].to(device)
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].to(device)
test_rec_images = test_rec_images[:,None,:,:].to(device)
test_images = test_images[:,None,:,:].to(device)

# indices = np.random.permutation(test_rec_images.shape[0])[:75]
f_test_images = test_images[list_of_test_images]
g_test_sinograms = test_noisy_sinograms[list_of_test_images]
f_test_rec_images = test_rec_images[list_of_test_images]

### Plotting one image from all and its sinogram and noisy sinogram
image_number = 50
noisy_sino = noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.subplot(1,3,2)
plt.imshow(noisy_sino)
plt.subplot(1,3,3)
plt.imshow(orig_sino)
plt.show()

image_number = 25
noisy_sino = test_noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = test_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.subplot(1,3,2)
plt.imshow(noisy_sino)
plt.subplot(1,3,3)
plt.imshow(orig_sino)
plt.show()


### Setting UNet as model and passing it to the used device
LGS_network = LGD(adjoint_operator_module, ray_transform_module, in_channels=2, out_channels=1, step_length=0.005, n_iter=5).to(device)

### Getting model parameters
LGS_parameters = list(LGD.parameters())

### Defining PSNR function
def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr

loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, f_images, g_sinograms, f_rec_images, f_test_rec_images, \
                  f_test_images, g_test_sinograms, n_train=50000, batch_size=4):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGS_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)

    ### Here starts the itarting in training
    for i in range(n_train):
        
        # scheduler.step()
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(g_sinograms.shape[0])[:batch_size]
        g_batch = g_sinograms[n_index,:,:,:]
        f_batch = f_images[n_index]
        f_batch2 = f_rec_images[n_index]
        
        net.train()
        
        optimizer.zero_grad()

        ### Evaluating the network which produces reconstructed images.
        outs, _ = net(f_batch2, g_batch)
        
        ### Calculating loss of the outputs
        loss = loss_train(f_batch, outs)
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(LGD_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        scheduler.step()

        ### Here starts the running tests
        if i % 100 == 0:
            
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            # net.eval()
            with torch.no_grad():
                outs2, step_len = net(f_test_rec_images, g_test_sinograms)
                ### Calculating test loss with test data outputs
                test_loss = loss_test(f_test_images, outs2).item()
            train_loss = loss.item()
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            
            ### Printing some data out
            if i % 500 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss):.2f}') #, end='\r'
                print(f'Step lenght: {step_len[0]}')
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs2[54,0,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(f_test_images[54,0,:,:].cpu().detach().numpy())
                plt.show()
                
    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
    plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(LGD, f_images, \
                                                                  g_sinograms, f_rec_images, \
                                                                  f_test_rec_images, f_test_images, \
                                                                  g_test_sinograms, n_train=50001, \
                                                                  batch_size=1)

### Evaluating the network
net.eval()
### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test_sinograms.shape[0], size=1))
LGD_reconstruction, _ = net(f_test_rec_images[None,image_number,:,:,:], g_test_sinograms[None,image_number,:,:,:])
LGD_reconstruction = LGD_reconstruction[0,0,:,:].cpu().detach().numpy()
ground_truth = f_test_images[image_number,0,:,:].cpu().detach().numpy()
noisy_reconstruction = f_test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_reconstruction)
plt.subplot(1,3,2)
plt.imshow(LGD_reconstruction)
plt.subplot(1,3,3)
plt.imshow(ground_truth)
plt.show()

# torch.save(net.state_dict(), '/scratch2/antti/networks/'+'LGS1_005.pth')

