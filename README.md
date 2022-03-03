# DeepForSpeed: A self-driving car in Need For Speed: Most Wanted
## Why i built this project?
The inspiration comes from how Nvidia built a self-driving car with just a single convolutional
neural network instead of many fancy algorithms combined. Here my goal is to replicate 
the amazing results they've gotten but inside a game. But also i also tried to create a platform/interface 
in which different architectures can be tested relatively easily, so it can also be used as a benchmark.
I tried to build it flexible enough so that it can function sort of as a fun driving simulator (of course
not an accurate one) that you can test your own neural networks at and conduct some experiments.


## Things used
> Python 3.9
> 
> Pytorch 1.10
>
> Numpy
> 
> Matplotlib
> 
> Base architecture
> 
> <img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/nvidia_arch.png?raw=true" width=30% height=30% alt="Nvidia's architecture">


## How to use it
There is different ways to use it depending on what you want. Additional info can be found inside the scripts.

Using models

<img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/flowchart.jpg?raw=true" width=50% height=50% alt="flowchart">

Creating and processing data

<img src="https://github.com/edilgin/DeepForSpeed/blob/master/images/dataFlowchart.jpg?raw=true" width=50% height=50% alt="flowchart">




## Pull requests

If you guys provide other Neural Network architectures they will be added. Also code can be refined in many ways that can make using it
simpler.


## For Future Updates:
> To see how networks perform add tensorflow board
> 
> Try saving images as rgb and see if it makes a difference
> 
> Add data augmentation functions
> 
> Try training with slight imbalanced data
> 
> Use a controller to get the input or a steering wheel
> 
> Different activation functions

### References:
paper by nvidia: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Sentdex's PyGta5 playlist: https://www.youtube.com/watch?v=ks4MPfMq8aQ&list=PLQVvvaa0QuDeETZEOy4VdocT7TOjfSA8a

