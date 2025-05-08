# Multilayer Perceptron Example

This project is an example for the MLP machine learning algorithm. It can train on the `MNIST_784` dataset and can also use it to predict the number a user draws.

The project comes packaged with a GUI you can use to check the prediction and the prediction calculated on my machine.

## Requirements
- `Python v3.6` (or above) - this project uses type/variable annotation for clarity. The code itself was written and ran using Python v3.13.
- Make sure you install the required modules listed in the `requirements.txt` file, they are:
  - `numpy` for our model's calculations.
  - `scikit-learn` for loading the `MNIST_784` dataset.
  - `tk` for opening a window.
  - `pillow` for processing images on the window.

## The Mathematics
We will be creating a 2 layer MLP as such:

$$
\text{input \_ layer} \longrightarrow \text{hidden \_ layer} \longrightarrow \text{output \_ layer}
$$

We'll now elaborate on what actually happens. Remainder, this is the flow for predicting a value, to learn, we'll have to apply a loss function on the output and use backwards propagation to calculate our gradients and optimize our weights.

### Notation

Our input will be represented as the matrix $\mathrm{X}\in\mathbb{R}^{m\times d}$ (in our case d=784 and m is the number of samples).

Each of our samples can be classified into $K=10$ classes.

Our weights are going to be $\mathrm{W}^{\left(1\right)}\in\mathbb{R}^{d\times h}$ and $\mathrm{b}^{\left(1\right)}\in\mathbb{R}^{1\times h}$ for the pass from the input layer to the hidden layer and $\mathrm{W}^{\left(2\right)}\in\mathbb{R}^{h\times K}$ and $\mathrm{b}^{\left(2\right)}\in\mathbb{R}^{1\times K}$ for the pass from the hidden layer to the output layer.

We'll use the sigmoid function as our first non-linear activation function and softmax as our second non-linear activation function to get our probabilities vector (we can then decode it into one-hot to find out the actual class our model predicted).

*Reminder*: The sigmoid and softmax functions are defined as such:
$$
\forall z\in\mathbb{R}\colon\sigma\left(z\right)=\frac{1}{1+e^{-z}}
$$
$$
\forall\mathrm{z}\in\mathbb{R}^{K}\forall1\le i\le K\colon\left(\mathrm{softmax}\left(\mathrm{z}\right)\right)_{i}=\frac{e^{z_{i}}}{\sum_{k=1}^{K}e^{z_{k}}}
$$
Where, $v_i$ is the $i$th entry in the vector $\mathrm v$. The softmax function returns a vector.

### Activation

*Note*: Activating a function $f\colon \mathbb R \longrightarrow \mathbb R$ on a vector behaves as activating the function on each of the vector components. Thus we can activate the sigmoid function on a vector the get a sigmoid vector.

For the pass from the input to the hidden layer we'll have the following pre-activation and activation:

$$
\mathrm{Z}^{\left(1\right)}=\mathrm{X}\mathrm{W}^{\left(1\right)}+\mathrm{b}^{\left(1\right)},
\hspace{3em}
\mathrm{A}^{\left(1\right)}=\sigma\left(\mathrm{Z}^{\left(1\right)}\right)\hspace{3em}
\in\mathbb{R}^{m\times h}
$$

For the second pass we are doing something similar

$$
\mathrm{Z}^{\left(2\right)}=\mathrm{A}^{\left(1\right)}\mathrm{W}^{\left(2\right)}+\mathrm{b}^{\left(2\right)},
\hspace{3em}
\mathrm{A}^{\left(2\right)}=\mathrm{softmax}\left(\mathrm{Z}^{\left(1\right)}\right)\hspace{3em}\in\mathbb{R}^{m\times K}
$$

where $\mathrm{Row}_{i}\left(\mathrm{A}^{\left(2\right)}\right)=\mathrm{softmax}\left(\mathrm{Row}_{i}\left(\mathrm{Z}^{\left(2\right)}\right)\right)$ ($\mathbb A ^{\left(2\right)}$ is a column vector).

*Note*: We are adding a matrix to a vector. Because our $\mathrm b$ vectors are a row vector, the addition is defined as adding $\mathbb b$ to each of the rows. This is the behavior numpy defines (and a lot of other computational modules for a just reason) and this is how we'll define it.

### Output
We'll have our labels be represented as $\mathrm Y \in \left\{0, 1\right\}^{m\times K}$ where each row is a **one-hot** representation of the true label.

### Loss
We will be using the *Cross-Entropy Loss* which looks like:

$${\cal L}=-\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{K}\mathrm{Y}_{i,j}\ln\mathrm{A}_{i,j}^{\left(2\right)}$$

### Backwards Propagation
#### Derivatives w.r.t. $\mathrm Z^{\left(1\right)}$
For a single sample i, we can define our loss to be:

$${\cal L}=-\sum_{k=1}^{K}y_{k}\ln a_{k},\hspace{3em}a_{k}=\left(\mathrm{softmax}\left(\mathrm{z}\right)\right)_{k}$$

First,

$$\frac{\partial{\cal L}}{\partial a_{k}}=-\frac{y_{k}}{a_{k}}$$

Next, we calculate the Jacobian of softmax to get

$$\frac{\partial a_{k}}{\partial z_{j}}=a_{k}\left(\delta_{kj}-a_{j}\right)$$

where $\delta_{kj}=0$ if $k\neq j$ and $\delta_{kj}=1$ if $k=j$.

Now get get that:

$$\frac{\partial{\cal L}}{\partial z_{j}}=\sum_{k=1}^{K}\frac{\partial{\cal L}}{\partial a_{k}}\frac{\partial a_{k}}{\partial z_{j}}=-\sum_{k=1}^{K}\frac{y_{k}}{a_{k}}a_{k}\left(\delta_{kj}-a_{j}\right)=-\sum_{k=1}^{K}y_{k}\delta_{kj}+\sum_{k=1}^{K}y_{k}a_{j}=a_{j}-y_{j}$$

as $a_{k}$ cancels out and $\delta_{kj}$ gives us 1 only when $k=j$ and $y_{k}=1$ only when $j=k$ (as per our one-hot encoding).

And for all $m$ samples we get

$$
\boxed{
  \frac{\partial\cal L}{\partial\mathrm Z^{\left(2\right)}} = \frac{1}{m}\left(\mathrm A^{\left(2\right)} - \mathrm Y\right)
}
$$

#### Gradients for $\mathrm W^{\left(2\right)},\mathrm b^{\left(2\right)}$

Using

$$\mathrm{Z}^{\left(2\right)}=\mathrm{A}^{\left(1\right)}\mathrm{W}^{\left(2\right)}+\mathrm{b}^{\left(2\right)}$$

$$\frac{\partial\mathrm{Z}^{\left(2\right)}}{\partial\mathrm{W}^{\left(2\right)}}=\mathrm{A}^{\left(1\right)}\hspace{3em}\frac{\partial\mathrm{Z}^{\left(2\right)}}{\partial\mathrm{b}^{\left(2\right)}}=I_{h\times K}$$

Thus, we get

$$
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{W}^{\left(2\right)}}=\left(\mathrm{A}^{\left(1\right)}\right)^{\top}\cdot\frac{\partial{\cal L}}{\partial\mathrm{Z}^{\left(2\right)}}
}
\hspace{3em}
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{b}^{\left(2\right)}}=\sum_{i=1}^{m}\frac{\partial{\cal L}}{\partial\mathrm{Z}_{i}^{\left(2\right)}}
}
$$

Where $\mathrm{Z}_{i}^{\left(2\right)}$ is the ith row.

#### Backprop into layer 1

Propagate through weights

$$
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{A}^{\left(1\right)}}=\frac{\partial{\cal L}}{\partial\mathrm{Z}^{\left(2\right)}}\cdot\left(\mathrm{W}^{\left(2\right)}\right)^{\top}
}
$$

With $\sigma'\left(z\right)=\sigma\left(z\right)\left(1-\sigma\left(z\right)\right)$ we get

$$
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{Z}^{\left(1\right)}}=\frac{\partial{\cal L}}{\partial\mathrm{A}^{\left(1\right)}}\odot\sigma'\left(\mathrm{Z}^{\left(1\right)}\right)
}
$$

Where $\odot$ is element-wise multiplication.

#### Gradients for $\mathrm W^{\left(1\right)},\mathrm b^{\left(1\right)}$

Since $\mathrm{Z}^{\left(1\right)}=\mathrm{X}\mathrm{W}^{\left(1\right)}+\mathrm{b}^{\left(1\right)}$ we get that

$$
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{W}^{\left(1\right)}}=\mathrm{X}^{\top}\frac{\partial{\cal L}}{\partial\mathrm{X}^{\top}}
}
\hspace{3em}
\boxed{
  \frac{\partial{\cal L}}{\partial\mathrm{b}^{\left(1\right)}}=\sum_{i=1}^{m}\frac{\partial{\cal L}}{\partial\mathrm{Z}_{i}^{\left(1\right)}}
}
$$

#### Parameter update
With our learning rate being $\eta$ we get:

$$
\mathrm{W}^{\left(2\right)}\gets\mathrm{W}^{\left(2\right)}-\eta\frac{\partial{\cal L}}{\partial\mathrm{W}^{\left(2\right)}}
\hspace{3em}
\mathrm{b}^{\left(2\right)}\gets\mathrm{b}^{\left(2\right)}-\eta\frac{\partial{\cal L}}{\partial\mathrm{b}^{\left(2\right)}}
$$

$$
\mathrm{W}^{\left(1\right)}\gets\mathrm{W}^{\left(1\right)}-\eta\frac{\partial{\cal L}}{\partial\mathrm{W}^{\left(1\right)}}
\hspace{3em}
\mathrm{b}^{\left(1\right)}\gets\mathrm{b}^{\left(1\right)}-\eta\frac{\partial{\cal L}}{\partial\mathrm{b}^{\left(1\right)}}
$$
