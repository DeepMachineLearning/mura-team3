# mura-team3
## MURA update
### Recommendation for Computing Power
CPU: 8 Cores - 12 Cores

CPU Memory: >= 70G (load all images with format 224x224x3)

GPU: 2

GPU Memory: >=10G per GPU

### Data Pre-process
1. Change "data_path" variable in mura_dataload.py file to your path that holds "MURA-V1.1" train/validation files.  
   Then Run: "python mura_dataload.py" 
   
### Train the model

Run: "python IncV3.py --epochs 20"
If you want to try other hpyerparameters, please run "python IncV3.py --help"


## Week 1 Task: MNIST Training

## Getting Started

The following instructions is based on my experience with AWS AMI setup.

### Prerequisites 

You need to have AWS account in order to setup AWS Deep Learning AMI.

### Installing

Task 1: create a EC2 instance with DL AMI. If you use Windows, please ignore step 3a from the link below and perform task 2 first. Then you can resume step 3b to start the jupyter notebook.

#### https://aws.amazon.com/getting-started/tutorials/get-started-dlami/

Task 2: Connecting to Your DL AMI Instance from Windows Using PuTTY

#### https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html

## Running the MNIST Keras code by using jupyter

myMNIST.ipynb

## Learning CNN resources

#### http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html

## Authors

* **Walter Lu** - *Initial work* 
