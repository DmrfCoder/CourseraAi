# Planar data classification with one hidden layer

From [Logistic Regression with a Neural Network mindset](https://blog.csdn.net/qq_36982160/article/details/83934571), we achieved the Neural Network which use Logistic Regression to resolve the linear classification . In this blog ,we will achieve a Neural Network with one hidden layer to resolve the no-linear classification as :

![image-20181113084012793](https://ws2.sinaimg.cn/large/006tNbRwgy1fx652rr46lj30sw0k2gsi.jpg)

## Which I will Code

- Implement a 2-class classification neural network with a single hidden layer

- Use units with a non-linear activation function, such as tanh 

- Compute the cross entropy loss 

- Implement forward and backward propagation

## Defining the neural network structure 

### layer_size()

This function will define three variables:

- n_x: the size of the input layer
- n_h: the size of the hidden layer (set this to 4) 

- n_y: the size of the output layer

```python

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
   
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0]
    # size of output layer
    return (n_x, n_h, n_y)
```

## Initialize the model's parameters 

### initialize_parameters()

- To make sure our parameters' sizes are right. Refer to the neural network figure above if needed.
- I will initialize the weights matrices with random values. 
  - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).

- I will initialize the bias vectors as zeros. 
  - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.



```python
def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that our output matches ours although the initialization is random.
    
    W1 =  np.random.randn(n_h,n_x) * 0.01 
    b1 = np.zeros((n_h,1))
    W2 =  np.random.randn(n_y,n_h) * 0.01 
    b2 = np.zeros((n_y,1))

    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```

##  The Loop

### forward_propagation()

#### Step

  1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
2. Implement Forward Propagation. Compute $Z^{[1]}, A^{[1]}, Z^{[2]}$ and $A^{[2]}$ (the vector of all our predictions on all the examples in the training set).

3. Values needed in the backpropagation are stored in "`cache`". The `cache` will be given as an input to the backpropagation function.

#### Code

```python

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
   
    W1 = parameters['W1']
    b1 =  parameters['b1']
    W2 =  parameters['W2']
    b2 =  parameters['b2']
   
    
    # Implement Forward Propagation to calculate A2 (probabilities)
  
   
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
  
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

```

### compute_cost()

Now that I have computed A^{[2]} (in the Python variable "`A2`"), which contains $a^{[2](i)}$ for every example, I can compute the cost function as follows:
$$
J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large{(} \small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right) \large{)} \small\tag{1}
$$


```python

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (1)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y,np.log(A2))+np.multiply((1-Y),np.log(1-A2))
    cost = -np.sum(logprobs)/m
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost
```

 ###  backward propagation()

Backpropagation is usually the hardest (most mathematical) part in deep learning.  Here is the slide from the lecture on backpropagation. I'll want to use the six equations on the right of this slide, since I are building a vectorized implementation.  

![grad_summary](https://ws4.sinaimg.cn/large/006tNbRwgy1fx65pv66zgj31kw0ufwwp.jpg)


$\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } = \frac{1}{m} (a^{[2](i)} - y^{(i)})$

$\frac{\partial \mathcal{J} }{ \partial W_2 } = \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } a^{[1] (i) T}$

$\frac{\partial \mathcal{J} }{ \partial b_2 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)}}}$

$\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} } =  W_2^T \frac{\partial \mathcal{J} }{ \partial z_{2}^{(i)} } * ( 1 - a^{[1] (i) 2}) $

$\frac{\partial \mathcal{J} }{ \partial W_1 } = \frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)} }  X^T $

$\frac{\partial \mathcal{J} _i }{ \partial b_1 } = \sum_i{\frac{\partial \mathcal{J} }{ \partial z_{1}^{(i)}}}$

- Note that $*$ denotes elementwise multiplication.
- The notation I will use is common in deep learning coding:
    - dW1 = $\frac{\partial \mathcal{J} }{ \partial W_1 }$
    - db1 = $\frac{\partial \mathcal{J} }{ \partial b_1 }$
    - dW2 = $\frac{\partial \mathcal{J} }{ \partial W_2 }$
    - db2 = $\frac{\partial \mathcal{J} }{ \partial b_2 }$



- Tips:
    - To compute dZ1 we need to compute $g^{[1]'}(Z^{[1]})$. Since $g^{[1]}(.)$ is the tanh activation function, if $a = g^{[1]}(z)$ then $g^{[1]'}(z) = 1-a^2$. So we can compute 
      $g^{[1]'}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.

```python

def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters['W1']
    W2 = parameters['W2']
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### (≈ 2 lines of code)
    A1 = cache['A1']
    A2 = cache['A2']
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

```

**General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.

**Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.

![sgd](https://ws1.sinaimg.cn/large/006tNbRwgy1fx65tkcmsmg30ao080my0.gif )![sgd_bad](https://ws4.sinaimg.cn/large/006tNbRwgy1fx65tp8t8cg30ao080gn2.gif)

if the learning rate is fit, the training gradient will descent as the left Gif, While if we use a too bad learning rate ,the gradient will descent like the right Gif.

```python

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
   
    W1 = parameters['W1']
    b1 =parameters['b1']
    W2 = parameters['W2']
    b2 =parameters['b2']
  
    
    # Retrieve each gradient from the dictionary "grads"
   
    dW1 = grads["dW1"]
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
  
    
    # Update rule for each parameter
   
    W1 = W1-learning_rate*dW1
    b1 =  b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 =  b2-learning_rate*db2
   
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

```

## Integrate above base function in nn_model()

```python

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
   
    parameters = initialize_parameters(X.shape[0],n_h,Y.shape[0])
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
 
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
       
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X,parameters)
        
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2,Y,parameters)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters,cache,X,Y)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters,grads)
        
       
        
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


```

## Predictions

Now I will  use our model to predict by building predict().
Use forward propagation to predict results.

**Reminder**: predictions = $y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
​      1 & \text{if}\ activation > 0.5 \\
​      0 & \text{otherwise}
​    \end{cases}$  

As an example, if we would like to set the entries of a matrix X to 0 and 1 based on a threshold we would do: ```X_new = (X > threshold)```

```python

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
 
    A2, cache = forward_propagation(X,parameters)
    predictions = (A2>0.5)
   
    
    return predictions
```

It is time to run the model and see how it performs on a planar dataset:

```python
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

```

Output;

![image-20181113092455705](https://ws2.sinaimg.cn/large/006tNbRwgy1fx66cw3zrjj310y0xwwns.jpg)

<table style="width:40%">
  <tr>
    <td>Cost after iteration 9000</td>
    <td> 0.218607 </td> 
  </tr>
</table>

#### Print accuracy

```python
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
```

Output:

<table style="width:15%">
  <tr>
    <td>Accuracy</td>
    <td>90% </td> 
  </tr>
</table>

## Tuning hidden layer size (optional/ungraded exercise)

```python

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


```

Output:

![image-20181113093334858](https://ws1.sinaimg.cn/large/006tNbRwgy1fx66lz1aerj31kg0tgn8u.jpg)

![image-20181113093347421](https://ws3.sinaimg.cn/large/006tNbRwgy1fx66m84uglj31kg0lmk1n.jpg)

![image-20181113093408391](https://ws1.sinaimg.cn/large/006tNbRwgy1fx66midno9j31kg0lmakg.jpg)

![image-20181113093420924](https://ws4.sinaimg.cn/large/006tNbRwgy1fx66mp0303j31kg0lmq8y.jpg)



### Interpretation

- The larger models (with more hidden units) are able to fit the training set better, until eventually the largest models overfit the data. 
- The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to  fits the data well without also incurring noticable overfitting.
- You will also learn later about regularization, which lets you use very large models (such as n_h = 50) without much overfitting. 