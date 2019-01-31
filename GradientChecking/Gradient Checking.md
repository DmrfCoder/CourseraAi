# Gradient Checking

## Abstract

Gradient Checking is a important method to verify whether our backpropagation code is correct or not. In this article, I will introduce the theory of n-layer gradient checking firstly, and then i will implement a 3 layer gradient checking step by step,hope after this article,you can implement the gradient checking method to resolve your neural network's backpropagation bugs.

## Theory

We know we can use $\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$ to estimate the grad of J in $\theta$ :
$$
\lim\limits_{\epsilon \to0}\frac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}\approx\frac{\alpha{J}}{\alpha{\theta}}
$$


But for the n layer neural network ,there are many parameters as $W^{[1]},b^{[1]},W^{[2]},b^{[2]},...,W^{[l]},b^{[l]}$, and $dW^{[1]},db^{[1]},dW^{[2]},db^{[2]},...,dW^{[l]},db^{[l]}$ for these parameters when we do backpropagation, we can merge these parameters to a big parameter named $\theta$  wich contains $\theta^{[1]},\theta^{[2]},..,\theta^{[l]}$and $d\theta$ contains  $d\theta^{[1]},d\theta^{[2]},..,d\theta^{[l]}$. For gradient checking,we can fix the number of $l-1$ parameters and assume that $J$ is only related with one parameter $\theta^{[i]}$ ,so we can estiimate the grad of J in $\theta^{[i]}$:
$$
d\theta_{approx}^{[i]}=\lim\limits_{\epsilon \to0}\frac{J(\theta^{[i]}+\epsilon)-J(\theta^{[i]}-\epsilon)}{2\epsilon}\approx\frac{\alpha{J}}{\alpha{\theta^{[i]}}}
$$


Thus we can coculate $d\theta_{approx}^{[i]}$ for every $\theta^{[i]}$  and then we should measure whether $d\theta_{approx}$ is close $d\theta$ within our acceptable range by this formla :
$$
difference=\frac{||d\theta_{approx}-d\theta||_2}{||d\theta_{approx}||_2+||d\theta||_2}
$$


I will set $\epsilon=1e^{-7}$ and if $difference \le 1e^{-7}$ ,we can say our backpropagation is correct , if  $1e^{-7} < difference\le 1e^{-5}$ ,we should alert whether our backpropagation is correct,and if $difference>1e^{-5}$

there is a great possibility that our backpropagation is incorrect.



## Implement

First of all, we need implement the `dictionary_to_vector` and `vector_to_dictionary` function to convert the parameters with parameter:

### dictionary_to_vector

```python
def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys
```

### vector_to_dictionary

```python

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters
```

Then we can implements the forward and backward propagation of neural network:

### forward_propagation_n

```python


def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.

    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    cost -- the cost function (logistic cost for one example)
    """

    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache
```

### backward_propagation_n

```python

def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()

    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

```



then we can implement the `gradient_check`:

### gradient_check

#### Instructions

Here is pseudo-code that will help you implement the gradient check.

For each i in num_parameters:
- To compute `J_plus[i]`:
    1. Set $\theta^{+}$ to `np.copy(parameters_values)`
    2. Set $\theta^{+}_i$ to $\theta^{+}_i + \varepsilon$
    3. Calculate $J^{+}_i$ using to `forward_propagation_n(x, y, vector_to_dictionary(`$\theta^{+}$ `))`.     
- To compute `J_minus[i]`: do the same thing with $\theta^{-}$
- Compute $gradapprox[i] = \frac{J^{+}_i - J^{-}_i}{2 \varepsilon}$

Thus, we get a vector gradapprox, where gradapprox[i] is an approximation of the gradient with respect to `parameter_values[i]`. You can now compare this gradapprox vector to the gradients vector from backpropagation. Just like for the 1D case (Steps 1', 2', 3'), compute: 
$$ difference = \frac {\| grad - gradapprox \|_2}{\| grad \|_2 + \| gradapprox \|_2 } \tag{3}$$




```python

def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n
    
    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters. 
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)
    
    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """
    
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    
    # Compute gradapprox
    for i in range(num_parameters):
        
         # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
       
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] = thetaplus[i] + epsilon  # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus))  # Step 3
    

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
       
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] = thetaminus[i] - epsilon  # Step 2
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus))  # Step 3
       

        # Compute gradapprox[i]
       
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)
      
    # Compare gradapprox to backward propagation gradients by computing difference.
   
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'
  

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference
```

### test gradient checking




```python
X, Y, parameters = gradient_check_n_test_case()

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)
```

Output:

![image-20181214113603114](https://ws1.sinaimg.cn/large/006tNbRwgy1fy64cuy51xj315a01i0sy.jpg)



It seems that there were errors in the `backward_propagation_n` code we implement! Good that we've implemented the gradient check. Go back to `backward_propagation` and try to find/correct the errors :

```python
dW2 = 1. / m * np.dot(dZ2, A1.T) * 2
db1 = 4. / m * np.sum(dZ1, axis=1, keepdims=True)
```

It should be:

```
dW2 = 1. / m * np.dot(dZ2, A1.T) 
db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)
```

then we run the test gradient cheking code:

![image-20181214114130311](https://ws3.sinaimg.cn/large/006tNbRwgy1fy64ijl06hj315a01iaa8.jpg)

the backpagation is correct now!

## What you should remember from this article



- Gradient checking verifies closeness between the gradients from backpropagation and the numerical approximation of the gradient (computed using forward propagation).
- Gradient checking is slow, so we don't run it in every iteration of training. You would usually run it only to make sure your code is correct, then turn it off and use backprop for the actual learning process. 

## Links

[github](https://github.com/DmrfCoder/CourseraAi/tree/master/GradientChecking)

