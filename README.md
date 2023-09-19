# Cambridge-Autumn-School
Jupyter Notebooks and needed files for Autumn School in University of Cambridge

To get the Jupyter Notebooks to work, one needs to install several packages for Python. <br>

Preferably first one installs [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).

After installation there is to create a new conda environment which will contain a neede Python version and all the packages.

The command to create new conda is `conda create --name yourenv`, where "yourenv" is the name of your environment.

Then one can activate the environment with command `conda activate yourenv`. (Different in Linux and Windows).

When the environment is activated, one can start to install everything. Next up is alot of lines to execute in your conda environment.

`conda install python=3.9`
`pip install https://github.com/odlgroup/odl/archive/master.zip`
From [PyTorch](https://pytorch.org/get-started/locally/)-site one can get the command which installs PyTorch
`pip install opencv-python-headless`
`conda install -c astra-toolbox astra-toolbox`
`conda install matplotlib`
`pip install ipykernel`

Lastly one needs to downgrade NumPy and SciPy:
`python -m pip install numpy==1.23.5`
`python -m pip install scipy==1.8.1`

