##Algorithms in this library have not been extensively tested; some likely have bugs
**William Agnew's Data Flow Graph Library**

This is a GPU accelerated Java data flow graph library in the same vein as TensorFlow and Theano. It contains all the differentiation, evaluation, and Java superclass/interface framework for you to start implementing, combining, and training your own differentiable functions and graphs! 

##Visualization
This library is integrated with <a href="http://graphstream-project.org/">GraphStream</a>, allowing you to visualize and debug your data flow graphs. Below on the left is a single LSTM unit; to the right of it is that LSTM unit unrolled 25 times.

<img src="https://github.com/wagnew/ComputeGraph/tree/master/ComputeGraph/data/LSTM.jpg" width="400"> 

<img src="https://github.com/wagnew/ComputeGraph/tree/master/ComputeGraph/data/LSTMU25.jpg" width="400">

##Implemented Algorithms
* Sigmoid, TanH, Add, Multiply, Ebe Multiply, Matrix Combine and Split Differentiable Vertices
* Cross Entropy and Euclidean Loss Functions
* AdaDelta, Adam, SGD, Nestrov Momentum, and RProp Optimizers
* Vanilla and LSTM Recurrent Neural Networks
* Basic Feedforward Neural Networks are Fairly Trvial to Implement

##Implementing Your Own Differentiable Functions

With the lessons learned from my first machine learning library (https://github.com/wagnew3/mlgpu), over about a week I created a dataflow graph library to implement more complex neural network archetectures