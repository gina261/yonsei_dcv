"""
Implements a two-layer neural network classifier in numpy.
"""
import numpy as np

class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for a linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.

        Inputs:
        - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
        - w: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: Output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in `out` #
        # Note that you need to reshape the input into rows.                 #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        D = 1
        for i in range(1, len((x.shape))):
          D *= x.shape[i]
        N = x.shape[0]
        x_reshape = x.reshape(N, D)
        one_matrix = np.ones((N,1))
        M = w.shape[1]
        b_reshape = b.reshape((1,M))
        # print(x_reshape.shape)
        # print(w.shape)
        # print(one_matrix.shape)
        # print(b_reshape.shape)
        
        out = x_reshape @ w + one_matrix @ b_reshape
        
        # print(out.shape)
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a linear (fully-connected) layer.

        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: A numpy array containing input data, of shape (N, d_1, ... d_k)
          - w: A numpy array of weights, of shape (D, M)
          - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ######################################################################
        # TODO: Implement the linear backward pass.                          #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        # dout = (dL / dy)
        N = x.shape[0]
        D = w.shape[0]
        M = b.shape[0]
                
        ones_matrix = np.ones((1,N))
        
        db = ones_matrix @ dout
        dw = (x.reshape(N,D)).T @ dout
        dx = dout @ w.T
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Compute the forward pass for a layer of rectified linear unit (ReLU).

        Input:
        - x: A numpy array containing input data, of any shape

        Returns a tuple of:
        - out: Output; a numpy array of the same shape as x
        - cache: x
        """
        out = None
        ######################################################################
        # TODO: Implement the ReLU forward pass.                             #
        # You should not change the input tensor with an in-place operation. #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        # max ( 0, x) 
        out = x.copy()
        out[x < 0] = 0
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Compute the backward pass for a layer of rectified linear unit (ReLU).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: A numpy array containing input data, of the same shape as dout

        Returns:
        - dx: Gradient with respect to x, of the same shape as dout
        """
        dx, x = None, cache
        ######################################################################
        # TODO: Implement the ReLU backward pass.                            #
        # You should not change the input array with an in-place operation.  #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        indicator_x = (x>0)
        dx = dout * indicator_x
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################
        return dx


def l2_loss(x, y):
    """
    Compute the loss and gradient for L2 loss.

    loss = .5 * sum_i (x_i - y_i)**2 / N

    Inputs:
    - x: A numpy array of shape (N, D) containing input data
    - y: A numpy array of shape (N, D) containing output data

    Returns a tuple of:
    - loss: a float containing the loss
    - dx: A numpy array of shape (N, D) containing the gradient of the loss
      with respect to x
    """
    N = len(x)
    diff = x - y
    loss = .5 * np.sum(diff * diff) / N
    dx = diff / N
    return loss, dx


def softmax_loss(x, y):
    """
    Compute the loss and gradient for softmax (cross-entropy) loss function.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - x: A numpy array of shape (N, C) containing predicted class scores;
      x[i,c] gives the predicted score for class c on i-th input sample
    - y: A numpy array of shape (N,) containing ground-truth labels;
      y[i] = c means that i-th input sample has label c, where 0 <= c < C.

    Returns a tuple of:
    - loss: a float containing the loss
    - dx: A numpy array of shape (N, C) containing the gradient of the loss
      with respect to x
    """
    loss, dx = None, None
    ##########################################################################
    # TODO: Compute the softmax loss and its gradient.                       #
    # Store the loss in loss and the gradient in dx. If you are not careful  #
    # here, it is easy to run into numeric instability; please refer to the  #
    # instruction.                                                           #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    # C is same as K class
    N = x.shape[0] 
    C = x.shape[1]
    s_max = np.max(x, axis =1).reshape((N,1))
    s_sum = np.sum(np.exp(x - s_max), axis = 1).reshape((N,1))
    sy = np.zeros((N,1))
    for i in range(N):
      sy[i,0] = x[i, y[i]]
    loss_sum = (s_max - sy) + np.log(s_sum)
    # N minibatches
    loss = (np.sum(loss_sum) / N)  
    
    # Calculate softmax loss
    
    dx = np.zeros((N,C))
    
    # prob_x_k : (N, C)
    # prob_sum : (N, 1)
    s_sum = np.sum(np.exp(x-s_max), axis = 1).reshape(N,1)
    dx = (np.exp(x - s_max) / s_sum)
    indicator_matrix = np.zeros((N,C))
    for i in range(N):
      indicator_matrix[i, int(y[i])] = 1
    dx = (dx - indicator_matrix)    
    # consider minibatches
    dx = (dx / N)
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss, dx


def l2_regularization(w, reg):
    """
    Compute loss and gradient for L2 regularization of a weight matrix:

    loss = (reg / 2) * sum_i w_i^2

    Where the sum ranges over all elements of w.

    Inputs:
    - w: A numpy array of any shape
    - reg: A scalar giving the regularization strength

    Returns:
    """
    loss, dw = None, None
    ##########################################################################
    # TODO: Implement L2 regularization.                                     #
    # NOTE: To ensure your implementation matches ours and you pass the      #
    # automated tests, make sure that your L2 regularization includes        #
    # a factor of 0.5 to simplify the expression for the gradient.           #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    sum_w = np.sum(w ** 2)
    loss = (reg * 0.5) * sum_w    
    
    # Backward path
    grad_sum_w = (reg *0.5) * 1.000
    # All of the same gradientes
    ones_matrix = np.ones(w.shape)
    grad_w = ones_matrix * grad_sum_w
    dw = 2 * w * grad_w
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return loss, dw


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input
    dimension of D, a hidden dimension of H, and perform classification over
    C classes.
    The architecture should be linear - relu - linear - softmax.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    def __init__(self, input_dim=1*28*28, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: A scalar giving the standard deviation of a Gaussian
          distribution for random initialization of the weights. The bias
          vectors of the model will always be initialized to zero.
        - reg: A scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ######################################################################
        # TODO: Initialize the weights and biases of the two-layer net.      #
        # Weights should be initialized from a Gaussian centered at 0.0 with #
        # the standard deviation equal to weight_scale, and biases should be #
        # initialized to zero. All weights and biases should be stored in    #
        # the dictionary self.params, with the first layer weights and       #
        # biases using the keys 'W1' and 'b1' and second layer weights and   #
        # biases using the keys 'W2' and 'b2'.                               #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["b2"] = np.zeros(num_classes)
        self.params["W1"] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params["W2"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def parameters(self):
        """
        Return a dictionary of all learnable parameters for this model.
        """
        return self.params

    def predict(self, X):
        """
        Make predictions for a batch of images.

        Inputs:
        - X: A numpy array of shape (N, D) giving input images to classify

        Returns:
        - y_pred: A numpy array of shape (N,) where each element is an integer
          in the range 0 <= y_pred[i] < C giving the predicted category for
          the input X[i].
        """
        scores = self.loss(X)
        y_pred = scores.argmax(axis=1)
        return y_pred

    def save(self, path):
        checkpoint = {
          'params': self.params,
          'reg': self.reg,
        }

        np.save(path, checkpoint, allow_pickle=True)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = np.load(path, allow_pickle=True).item()
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p]
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: A numpy array of input data of shape (N, d_1, ..., d_k)
        - y: A numpy array of labels, of shape (N,).
          y[i] is the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: A numpy array of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i] and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: A scalar value giving the loss
        - grads: A dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        ######################################################################
        # TODO: Implement the forward pass for the two-layer net, computing  #
        # the class scores for X and storing them in the scores variable.    #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        
        # Forward Path
        hid_y , cache_hid = Linear.forward(X, W1, b1)
        act_y , cache_act= ReLU.forward(hid_y)
        scores, cache_yhat = Linear.forward(act_y, W2, b2)
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ######################################################################
        # TODO: Implement the backward pass for the two-layer net.           #
        # Store the loss in the loss variable and gradients in the grads     #
        # dictionary. Compute data loss using softmax_loss, and make sure    #
        # that grads[k] holds the gradients for self.params[k].              #
        # Don't forget to add L2 regularization using l2_regularization!     #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        # Loss should be located behind
        loss, dL = softmax_loss(scores, y)
        
        # L2 regularization        
        pen1, dreg1 = l2_regularization(W1, self.reg)
        pen2, dreg2 = l2_regularization(W2, self.reg)
        
        # Backpropagation for backward
        dX2, dW2, db2 = Linear.backward(dL, cache_yhat)
        dR = ReLU.backward(dX2, cache_act)
        dX, dW1, db1 = Linear.backward(dR, cache_hid)
        
        # Add backward graients for regularizations
        grads["W2"] = dW2 + dreg2
        # grads["W2"] = dW2 
        grads["W1"] = dW1 + dreg1 
        # grads["W1"] = dW1
        grads["b2"] = db2
        grads["b1"] = db1
        
        # Add Regularization
        loss += (pen1 + pen2)
        
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

        return loss, grads


class SGD(object):
    """
    Perform stochastic gradient descent with momentum.

    - lr: A scalar learning rate.
    - momentum: A scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A dict of numpy arrays of the same shape as params used to
      store a moving average of the gradients.
    """
    def __init__(self, params, lr=1e-2, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def step(self, grads):
        if not self.velocity:
            for k in grads:
                self.velocity[k] = np.zeros_like(grads[k])
        ######################################################################
        # TODO: Implement the SGD update with momentum.                      #
        # You should update both self.params and self.velocity               #
        ######################################################################
        # Replace "pass" statement with your code (do not modify this line)
        
        for grad in grads:
          self.params[grad] = self.params[grad] -  self.velocity[grad] * self.lr
          self.velocity[grad] = self.momentum * self.velocity[grad] + grads[grad]
          
        ######################################################################
        #                          END OF YOUR CODE                          #
        ######################################################################

