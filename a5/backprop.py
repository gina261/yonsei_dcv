import math


"""
Defines forward and backward passes through different computational graphs.

Students should complete the implementation of all functions in this file.
"""


def f1(x1, w1, x2, w2, b, y):
    """
    Computes the forward and backward pass through the computational graph f1
    from the pdf instruction.

    A few clarifications about the graph:
    - The subtraction node in the graph computes d = y_hat - y
    - The ^2 node squares its input

    Inputs:
    - x1, w1, x2, w2, b, y: Python floats

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    giving the derivative of the output L with respect to each input.
    """
    # Forward pass: compute loss
    L = None
    ##########################################################################
    # TODO: Implement the forward pass for the computational graph f1 shown  #
    # in the pdf instruction. Store the loss in the variable L.              #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    # Forward pass implementation
    
    a1 = x1 * w1
    a2 = x2 * w2
    yhat =  (a1 + a2 + b)
    d = (yhat - y)
    L = (d ** 2)
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    # Backward pass: compute gradients
    grad_x1, grad_w1, grad_x2, grad_w2 = None, None, None, None
    grad_b, grad_y = None, None
    ##########################################################################
    # TODO: Implement the backward pass for the computational graph f1 shown #
    # in the pdf instruction. Store the gradients for each input variable in #
    # the corresponding grad variables defined above.                        #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    grad_L = 1.00
    grad_d = grad_L * 2 * d
    
    grad_yhat = grad_d * 1
    grad_y = grad_d * (-1)
    
    grad_a1 = grad_yhat * 1
    grad_a2 = grad_yhat * 1
    grad_b = grad_yhat * 1
    
    grad_x1 = grad_a1 * w1
    grad_x2 = grad_a2 * w2
    grad_w1 = grad_a1 * x1
    grad_w2 = grad_a2 * x2 
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    grads = (grad_x1, grad_w1, grad_x2, grad_w2, grad_b, grad_y)
    return L, grads


def f2(x):
    """
    Computes the forward and backward pass through the computational graph f2
    from the pdf instruction.

    A few clarifications about this graph:
    - The "x2" node multiplies its input by the constant 2
    - The "+1" and "-1" nodes add or subtract the constant 1
    - The division node computes y = t / b

    Inputs:
    - x: Python float

    Returns a tuple of:
    - y: Python float
    - grads: A tuple (grad_x,) giving the derivative of the output y with
      respect to the input x
    """
    # Forward pass: Compute output
    y = None
    ##########################################################################
    # TODO: Implement the forward pass for the computational graph f2 shown  #
    # in the pdf instruction. Store the output in the variable y.            #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    # Forward pass
    
    d = x * 2
    e = math.exp(d)
    
    # Copy for same operator
    e0 = e
    e1 = e
    
    # Same e value
    t = (e0 - 1)
    b = (e1 + 1)
    
    y = t / b
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    # Backward pass: Compute gradients
    grad_x = None
    ##########################################################################
    # TODO: Implement the backward pass for the computational graph f2 shown #
    # in the pdf instruction. Store the gradients for each input variable in #
    # the corresponding grad variables defined above.                        #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    grad_y = 1.00
    grad_b = grad_y * (-t/(b**2))
    grad_t  = grad_y / b
    
    grad_e0 = grad_t
    grad_e1 = grad_b
    grad_e = grad_e0 + grad_e1
    
    grad_d = grad_e * (math.exp(d))    
    grad_x = grad_d * 2

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    return y, (grad_x,)


def f3(s1, s2, y):
    """
    Computes the forward and backward pass through the computational graph f3
    from the pdf instruction.

    A few clarifications about the graph:
    - The input y is an integer with y == 1 or y == 2; you do not need to
      compute a gradient for this input.
    - The division nodes compute p1 = e1 / d and p2 = e2 / d
    - The choose(p1, p2, y) node returns p1 if y is 1, or p2 if y is 2.

    Inputs:
    - s1, s2: Python floats
    - y: Python integer, either equal to 1 or 2

    Returns a tuple of:
    - L: Python scalar giving the output of the graph
    - grads: A tuple (grad_s1, grad_s2) giving the derivative of the output L
    with respect to the inputs s1 and s2.
    """
    assert y == 1 or y == 2
    # Forward pass: Compute loss
    L = None
    ##########################################################################
    # TODO: Implement the forward pass for the computational graph f3 shown  #
    # in the pdf instruction. Store the loss in the variable L.              #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    e1 = math.exp(s1)
    e2 = math.exp(s2)    
    # Copy values
    e1_0 = e1
    e1_1 = e1
    e2_0 = e2
    e2_1 = e2
    
    d = e1_0 + e2_0
    # Copy values
    d0 = d
    d1 = d
    
    p1 = (e1_1 / d0)
    p2 = (e2_1 / d1)
    
    if y == 1 :
        pPlus = p1
    elif y == 2 :
        pPlus = p2
    else:
        print('Error')
        pPlus = 0
    
    L = -math.log(pPlus)
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    # Backward pass: Compute gradients
    grad_s1, grad_s2 = None, None
    ##########################################################################
    # TODO: Implement the backward pass for the computational graph f3 shown #
    # in the pdf instruction. Store the gradients for each input variable in #
    # the corresponding grad variables defined above. You do not need to     #
    # compute a gradient for the input y.                                    #
    # HINT: You may need an if statement to backprop through the chosen node #
    ##########################################################################
    # Replace "pass" statement with your code (do not modify this line)
    
    # Backpropagation
    grad_L = 1.00
    grad_pPlus = grad_L * (-1/pPlus)
    
    if y == 1 :
        grad_p2 = 0.00
        grad_p1 = grad_pPlus * 1 
    elif y == 2 :
        grad_p1 = 0.00
        grad_p2 = grad_pPlus * 1 
        
    grad_d0 = grad_p1 * (-e1_1/(d0**2))
    grad_d1 = grad_p2 * (-e2_1/(d1**2))
    # Copy operator
    grad_d = grad_d0 + grad_d1
    
    grad_e1_1 = grad_p1 / d0
    grad_e2_1 = grad_p2 / d1
    grad_e1_0 = grad_d
    grad_e2_0 = grad_d
    
    # Gather copied graident
    grad_e1 = grad_e1_0 + grad_e1_1
    grad_e2 = grad_e2_0 + grad_e2_1
    
    grad_s1 = grad_e1 * math.exp(s1)
    grad_s2 = grad_e2 * math.exp(s2)
    
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################

    grads = (grad_s1, grad_s2)
    return L, grads


def f3_y1(s1, s2):
    """
    Helper function to compute f3 in the case where y = 1

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=1)


def f3_y2(s1, s2):
    """
    Helper function to compute f3 in the case where y = 2

    Inputs:
    - s1, s2: Same as f3

    Outputs: Same as f3
    """
    return f3(s1, s2, y=2)

