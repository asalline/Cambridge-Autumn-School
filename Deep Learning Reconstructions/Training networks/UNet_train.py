### Post-processing image enhancement with Filtered Back Projection using UNet

###
### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -matplotlib
###                  -UNet_functions.py (NEEDS ITS OWN PACKAGES EG. OpenCV)
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
from UNet_train_module import get_images, geometry_and_ray_trafo, UNet
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()

### Using function "get_images" to import images from the path.
images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images='all', scale_number=2)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)

### Using functions from "UNet_train_module". Taking shape from images to produce
### odl parameters and getting ray transform operator and its adjoint.
shape = (np.shape(images)[1], np.shape(images)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

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

### Adding Gaussian noise to the sinograms
for k in range(np.shape(sinograms)[0]):
    sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinogram = sinogram_k + noise
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

### All the data into same device
sinograms = sinograms[:,None,:,:].cpu().detach()
noisy_sinograms = noisy_sinograms[:,None,:,:].cpu().detach()
rec_images = rec_images[:,None,:,:].cpu().detach()
images = images[:,None,:,:].cpu().detach()

### Seperating the data into training and and testing data. 
### "g_" is data from reconstructed images and
### "f_" is data from ground truth images
g_train = rec_images[0:n_train]
f_train = images[0:n_train]

### Here happens test images downloading. They are treated the same way as the training ones.
test_images = get_images('/scratch2/antti/summer2023/test_walnut', amount_of_images='all', scale_number=2)

list_of_test_images = list(range(0,363,5))

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

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
# g_test = test_rec_images[indices]
# f_test = test_images[indices]

g_test = test_rec_images[list_of_test_images]
f_test = test_images[list_of_test_images]

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

### Setting UNet as model and passing it to the used device
UNet_network = UNet(in_channels=1, out_channels=1).to(device)

### Getting model parameters
UNet_parameters = list(UNet.parameters())

### Defining PSNR function.
def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr


loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

fbp_reco = fbp_operator(noisy_sinograms[50,0,:,:].cpu().detach().numpy())
plt.figure()
plt.imshow(fbp_reco)
plt.show()
print(psnr(loss_test(rec_images[50,0,:,:], images[50,0,:,:]).cpu().detach().numpy())**0.5)
# print('fbp_reco_shape', fbp_reco.size())
# print('image size', images[50,0,:,:].cpu().detach().numpy().size())
# print(psnr(loss_test(fbp_operator(noisy_sinograms[50,0,:,:].cpu().detach().numpy()), \
#                      images[50,0,:,:].cpu().detach().numpy())))

### Defining evaluation (test) function
def eval(net, g, f):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_test(net(g), f)).item())
    print(test_loss)
    out3 = net(g[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, n_train=300, batch_size=25): #g_train, g_test, f_train, f_test, 

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(UNet_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    ### Here starts the itarting in training
    for i in range(n_train):
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(g_train.shape[0])[:batch_size]
        g_batch = g_train[n_index,:,:,:].to(device)
        f_batch = f_train[n_index].to(device)
        
        
        net.train()

        ### Evaluating the network which not produces reconstructed images
        outs = net(g_batch)
        
        ### Setting gradient to zero
        optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(outs, f_batch)

        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(UNet_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        # scheduler.step()

        ### Here starts the running tests
        if i % 100 == 0:
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            outs2 = net(g_test)
            
            ### Calculating test loss with test data outputs
            test_loss = loss_test(outs2, f_test).item()
            train_loss = loss.item()
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            
            ### Printing some data out
            if i % 1000 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss):.2f}') #, end='\r'
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs2[54,0,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(f_test[54,0,:,:].cpu().detach().numpy())
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
running_loss, running_test_loss, net = train_network(UNet_network, n_train=50001, batch_size=1)

### Evaluating the network
net.eval()

### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test.shape[0], size=1))
UNet_reconstruction = net(g_test[None,image_number,:,:,:])[0,0,:,:].cpu().detach().numpy()
ground_truth = f_test[image_number,0,:,:].cpu().detach().numpy()
noisy_reconstruction = g_test[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_reconstruction)
plt.subplot(1,3,2)
plt.imshow(UNet_reconstruction)
plt.subplot(1,3,3)
plt.imshow(ground_truth)
plt.show()

# torch.save(net.state_dict(), '/scratch2/antti/networks/'+'UNet1_005.pth')

