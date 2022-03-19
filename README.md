# DeepForSpeed: Data Wanted
## A self-driving car in Need For Speed
The inspiration comes from how Nvidia built a self-driving car with just a single convolutional
neural network instead of many fancy algorithms combined. Here my goal is to replicate 
the amazing results they've gotten but inside a game. But i also tried to create it as a platform/interface 
in which different architectures can be tested relatively easily, so it can also be used as a benchmark.
So it's like a fun driving simulator (of course not an accurate one) that you can test your own neural networks 
at and maybe conduct some experiments.

## Watch The Latest Version
[![Watch the video](https://img.youtube.com/vi/t0iqfM36mRc/maxresdefault.jpg)](https://youtu.be/t0iqfM36mRc)
note: this is a cherry picked example and many times model will not perform this well. Im hoping to change that in future versions.

## Things used
> Python 3.9
> 
> Pytorch 1.10
>
> Numpy
> 
> OpenCV
> 
> Matplotlib
> 
> Need For Speed: Most Wanted 2005
> 
> Base architecture
> 
> <img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/nvidia_arch.png?raw=true" width=40% height=40% alt="Nvidia's architecture">


## How to use it
There is different ways to use it depending on what you want. Additional info can be found inside the scripts.


Creating and processing data

<img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/dataFlowchart.jpg?raw=true" width=50% height=50% alt="flowchart">

Using models

<img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/trainFlowchart.jpg?raw=true" width=50% height=50% alt="flowchart">


## Pull requests
TLDR: Basically any improvements are really appreciated.

- Other Neural Network architectures
- Refinements in the code
- Trained Models
- Anything you can get done on future updates part



## For Future Updates:
- Add tensorflow board
- Only use np arrays instead of both lists and np arrays in data
- RGB images instead of gray images
- Train on more data
- Increase data resolution
- Controller or a steering wheel to get the input
- Different activation functions
- Try Weight Decay
- Add merging data function for easing data creation
- Save models whilst training

### References:
paper by nvidia: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Sentdex's PyGta5 playlist: https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a

NFS:MW mods are taken from:  https://github.com/ExOptsTeam/NFSMWExOpts
