
import numpy as np


def t1(L):
    """
    Inputs:
    - L: A list of M numpy arrays, each of shape (1, N)

    Returns:
    A numpy array of shape (M, N) giving all inputs stacked together

    Par: 1 line
    Instructor: 1 line

    Hint: vstack/hstack/dstack, no for loop
    """
    
    return np.vstack(L)


def t2(X):
    """
    Inputs:
    - X: A numpy array of shape (N, N)

    Returns:
    Numpy array of shape (N,) giving the eigenvector corresponding to the
    smallest eigenvalue of X

    Par: 5 lines
    Instructor: 3 lines

    Hints:
    1) np.linalg.eig
    2) np.argmin
    3) Watch rows and columns!
    """
    value, vector = np.linalg.eig(X)
    return vector[:, np.argmin(value)]


def t3(X):
    """
    Inputs:
    - X: A numpy array of any shape

    Returns:
    A copy of X but with all negative entries set to 0

    Par: 3 lines
    Instructor: 2 lines

    Hints:
    1) np.copy
    2) If S is a boolean array with the same shape as X, then X[S] gives an
       array containing all elements of X corresponding to true values of S
    3) X[S] = v assigns the value v to all entries of X corresponding to
       true values of S.
    """
    X_copy = np.copy(X)
    X_copy[X < 0] = 0
    return X_copy


def t4(R, X):
    """
    Inputs:
    - R: A numpy array of shape (3, 3) giving a rotation matrix
    - X: A numpy array of shape (N, 3) giving a set of 3-dimensional vectors

    Returns:
    A numpy array Y of shape (N, 3) where Y[i] is X[i] rotated by R

    Par: 3 lines
    Instructor: 1 line

    Hints:
    1) If v is a vector, then the matrix-vector product Rv rotates the vector
       by the matrix R.
    2) .T gives the transpose of a matrix
    """    
    return np.dot(R, X.T).T


def t5(X):
    """
    Inputs:
    - X: A numpy array of shape (N, N)

    Returns:
    A numpy array of shape (4, 4) giving the upper left 4x4 submatrix of X
    minus the bottom right 4x4 submatrix of X.

    Par: 2 lines
    Instructor: 1 line

    Hint: X[y0:y1, x0:x1] gives the submatrix
          from rows y0 (inclusive) to y1 (exclusive)
          from columns x0 (inclusive) to x1 (exclusive)
    """
    N = X.shape[0]
    return X[0:4, 0:4] - X[N-4:N, N-4:N]


def t6(N):
    """
    Inputs:
    - N: An integer

    Returns:
    A numpy array of shape (N, N) giving all 1s, except the first and last 5
    rows and columns are 0.

    Par: 6 lines
    Instructor: 3 lines
    """
    nn = np.ones((N, N))
    nn[0:5] = 0
    nn[:,0:5] = 0
    nn[N-5:N] = 0
    nn[:, N-5:N] = 0
    return nn


def t7(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array Y of the same shape as X, where Y[i] is a vector that points
    the same direction as X[i] but has unit norm.

    Par: 3 lines
    Instructor: 1 line

    Hints:
    1) The vector v / ||v||| is the unit vector pointing in the same direction
       as v (as long as v != 0)
    2) Divide X by a vector of normalization factors, of shape (N, 1)
    3) Elementwise operations between an array of shape (N, M) and an array of
       shape (N, 1) work -- try it! This is called "broadcasting"
    4) Elementwise operations between an array of shape (N, M) and an array of
       shape (N,) won't work -- try reshaping
    """
    return X / np.linalg.norm(X,axis=1).reshape(-1, 1)


def t8(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array Y of shape (N, M) where Y[i] contains the same data as X[i],
    but normalized to have mean 0 and standard deviation 1.

    Par: 3 lines
    Instructor: 1 line

    Hint: Pay attention to the dimension (=axis)
          when calculating the mean and standard deviation.
    """ 
    # mean = np.sum(X, axis=1) / X.shape[1]
    mean = np.mean(X, axis =1)
    sd = np.std(X, axis = 1)
    result = (X - mean.reshape(-1,1)) / sd.reshape(-1,1)
    # sd = (np.sum((X - mean)**2) / X.shape[1])**0.5
    return result 


def t9(q, k, v):
    """
    Inputs:
    - q: A numpy array of shape (1, K) (queries)
    - k: A numpy array of shape (N, K) (keys)
    - v: A numpy array of shape (N, 1) (values)

    Returns:
    sum_i exp(-||q-k_i||^2) * v[i]

    Par: 3 lines
    Instructor: 1 ugly line

    Hints:
    1) You can perform elementwise operations on arrays of shape (N, K) and
       (1, K) with broadcasting
    2) Recall that np.sum has useful "axis" and "keepdims" options
    3) np.exp and friends apply elementwise to arrays
    """
    return np.sum(np.exp(-np.linalg.norm(q-k, axis=1, keepdims=True)**2) * v)


def t10(Xs):
    """
    Inputs:
    - Xs: A list of length L, containing numpy arrays of shape (N, M)

    Returns:
    A numpy array R of shape (L, L) where R[i, j] is the Euclidean distance
    between C[i] and C[j], where C[i] is an M-dimensional vector giving the
    centroid of Xs[i]

    Par: 12 lines
    Instructor: 3 lines (after some work!)

    Hints:
    1) You can use a for loop over L
    2) Distances are symmetric
    3) Go one step at a time
    4) Our 3-line solution uses no loops, and uses the algebraic trick from the
       next problem.
    """
    C = np.average(np.array(Xs), axis=1)
    norm_val = np.diag(C @ C.T)
    return np.sqrt(norm_val + norm_val.reshape((-1,1)) - 2 * C @ C.T)


def t11(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array D of shape (N, N) where D[i, j] gives the Euclidean distance
    between X[i] and X[j], using the identity
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y

    Par: 3 lines
    Instructor: 2 lines (you can do it in one but it's wasteful compute-wise)

    Hints:
    1) What happens when you add two arrays of shape (1, N) and (N, 1)?
    2) Think about the definition of matrix multiplication
    3) Transpose is your friend
    4) Note the square! Use a square root at the end
    5) On some machines, ||x||^2 + ||x||^2 - 2x^Tx may be slightly negative,
       causing the square root to crash. Just take max(0, value) before the
       square root. Seems to occur on Macs.
    """
    norm_val = np.diag(X @ X.T)
    result = np.sqrt(norm_val + norm_val.reshape((-1,1)) - 2 * X @ X.T)
    return result


def t12(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N, F)
    - Y: A numpy array of shape (M, F)

    Returns:
    A numpy array D of shape (N, M) where D[i, j] is the Euclidean distance
    between X[i] and Y[j].

    Par: 3 lines
    Instructor: 2 lines (you can do it in one, but it's more than 80 characters
                with good code formatting)

    Hints: Similar to previous problem
    """
    X_diag = np.diag(X @ X.T)
    Y_diag = np.diag(Y @ Y.T)
    return np.sqrt(X_diag.reshape(-1, 1) + Y_diag.reshape(1, -1) - 2 * X @ Y.T)


def t13(q, V):
    """
    Inputs:
    - q: A numpy array of shape (1, M) (query)
    - V: A numpy array of shape (N, M) (values)

    Return:
    The index i that maximizes the dot product q . V[i]

    Par: 1 line
    Instructor: 1 line

    Hint: np.argmax
    """
    return np.argmax(np.dot(q, V.T))


def t14(X, y):
    """
    Inputs:
    - X: A numpy array of shape (N, M)
    - y: A numpy array of shape (N, 1)

    Returns:
    A numpy array w of shape (M, 1) such that ||y - Xw||^2 is minimized

    Par: 2 lines
    Instructor: 1 line

    Hint: np.linalg.lstsq, or use the pseudoinverse (X^T X)^-1 X^T y
    """
    return np.linalg.lstsq(X,y,rcond=1)[0]


def t15(X, Y):
    """
    Inputs:
    - X: A numpy array of shape (N, 3)
    - Y: A numpy array of shape (N, 3)

    Returns:
    A numpy array C of shape (N, 3) such C[i] is the cross product between X[i]
    and Y[i]

    Par: 1 line
    Instructor: 1 line

    Hint: np.cross
    """
    return np.cross(X, Y)


def t16(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array Y of shape (N, M - 1) such that
    Y[i, j] = X[i, j] / X[i, M - 1]
    for all 0 <= i < N and all 0 <= j < M - 1

    Par: 1 line
    Instructur: 1 line

    Hints:
    1) If it doesn't broadcast, reshape or np.expand_dims
    2) X[:, -1] gives the last column of X
    """
    return (X / X[:,-1].reshape(-1,1))[:,:-1]


def t17(X):
    """
    Inputs:
    - X: A numpy array of shape (N, M)

    Returns:
    A numpy array Y of shape (N, M + 1) such that
        Y[i, :M] = X[i]
        Y[i, M] = 1

    Par: 1 line
    Instructor: 1 line

    Hint: np.hstack, np.ones
    """
    return np.hstack([X, np.ones((X.shape[0],1))])


def t18(N, r, x, y):
    """
    Inputs:
    - N: An integer
    - r: A floating-point number
    - x: A floating-point number
    - y: A floating-point number

    Returns:
    A numpy array I of floating point numbers and shape (N, N) such that:
    I[i, j] = 1 if ||(j, i) - (x, y)|| < r
    I[i, j] = 0 otherwise

    Par: 3 lines
    Instructor: 2 lines

    Hints:
    1) np.meshgrid and np.arange give you X, Y
    2) Arrays have an astype method
    """
    
    result = np.zeros((N,N))
    X, Y = np.meshgrid(np.arange(N), np.arange(N))
    result[((X-x)**2 + (Y-y)** 2) < r **2] = 1
    return result


def t19(N, s, x, y):
    """
    Inputs:
    - N: An integer
    - s: A floating-point number
    - x: A floating-point number
    - y: A floating-point number

    Returns:
    A numpy array I of shape (N, N) such that
    I[i, j] = exp(-||(j, i) - (x, y)||^2 / s^2)

    Par: 3 lines
    Instructor: 2 lines

    Hint: Be careful with types -- float and int aren't the same!
    """
    X_tot, Y_tot = np.meshgrid(np.arange(N), np.arange(N))
    I = np.exp(-((X_tot - x) ** 2 + (Y_tot - y) ** 2) / s**2)
    return I


def t20(N, v):
    """
    Inputs:
    - N: An integer
    - v: A numpy array of shape (3,) giving coefficients v = [a, b, c]

    Returns:
    A numpy array of shape (N, N) such that M[i, j] is the distance between the
    point (j, i) and the line a*j + b*i + c = 0

    Par: 4 lines
    Instructor: 2 lines

    Hints:
    1) The distance between the point (x, y) and the line ax+by+c=0 is given by
       abs(ax + by + c) / sqrt(a^2 + b^2)
       (The sign of the numerator tells which side the point is on)
    2) np.abs
    """
    X_tot, Y_tot = np.meshgrid(np.arange(N), np.arange(N))
    div = np.sqrt(np.sum(v[:2] ** 2))
    dist = np.abs(X_tot * v[0] + Y_tot * v[1] + v[2]) / div
    
    return dist
