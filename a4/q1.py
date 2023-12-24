import os
import numpy as np
from matplotlib import pyplot as plt

def fit_affine(X, Y):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit an affine transformation from [x,y] to [x',y'].

    Inputs:
        X: A numpy array of shape (N,2), each row is in the form [x_i,y_i]
        Y: A numpy array of shape (N,2), each row is in the form [x'_i,y'_i]

    Outputs:
        M: A linear transformation matrix of shape (2,2) that satisfies
        [x',y']^T = M [x,y]^T + t if fitting perfectly
        t: A translation vector of shape (2,) that satisfies
        [x',y']^T = M [x,y]^T + t if fitting perfectly
        Y_pred: A numpy array of shape (N,2), each row is in the form
        [x'_i,y'_i] that satisfies [x',y']^T = M [x,y]^T + t

    Hint: np.linalg.lstsq. You can reshape or break down the vector v of shape
    (6,) in the math for efficient implementation.
    '''
    M, t, Y_pred = None, None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    tot_point = X.shape[0]
    b = Y.reshape(2 * tot_point)
    
    A = np.array([[X[0][0], X[0][1], 0 , 0, 1, 0],
                       [0, 0, X[0][0], X[0][1], 0, 1 ]])
    
    for i in range(1, tot_point):
        append_np = np.array([[X[i][0], X[i][1], 0 , 0, 1, 0],
                       [0, 0, X[i][0], X[i][1], 0, 1 ]])
        A = np.concatenate([A, append_np], axis = 0)
    
    # print(A.shape)
    # print(b.shape)
    # print(X.shape)
    # print(Y.shape)
    
    sol = np.linalg.lstsq(A, b)[0]
    m0, m1, m2, m3, m4, m5 = sol
    
    M = np.array([[m0, m1],
                 [m2, m3]])
    t = np.array([[m4], [m5]])
    
    Y_pred = M @ X.T + t
    Y_pred = Y_pred.T
    
    ### END YOUR CODE   (do not modify this line)
    return M, t, Y_pred


def transform_homography(X, H):
    '''
    Perform homography transformation on a set of points X using homography H

    Inputs:
        X: A numpy array of shape (N,2), each row is in the form [x_i, y_i]
        H: A homography matrix of shape (3,3)

    Outputs:
        Y: A numpy array of shape (N,2) of the form [x', y'] satisfies
        [x',y',1]^T \equiv H [x,y,1]^T
    '''
    Xh = np.hstack([X,np.ones((X.shape[0],1))])
    Y = np.matmul(Xh, H.T)
    return Y[:,:2] / Y[:,2:]


def fit_homography(X, Y):
    '''
    Given a set of N correspondences X and Y of the form [x,y] and [x',y'],
    respectively, fit a homography from [x,y,1] to [x',y',1].

    Inputs:
        X: A numpy array of shape (N,2), each row is in the form [x_i,y_i]
        Y: A numpy array of shape (N,2), each row is in the form [x'_i,y'_i]

    Outputs:
        H: A homography matrix of shape (3,3) that satisfies H[-1,-1] = 1 and
        [x',y',1]^T \equiv H [x,y,1]^T if fitting perfectly

    Hint: np.linalg.svd or np.linalg.eig. You can rearrange the rows of
    the matrix of shape (2N,9) in the math for efficient implementation.
    You might also want to use np.where or np.nonzero.
    '''
    H = None
    ### YOUR CODE HERE  (do not modify this line)
    
    x = X[0][0]
    y = X[0][1]
    x_prime = Y[0][0]
    y_prime = Y[0][1]
    
    N = X.shape[0]
    
    # Set base A matrix
    
    A_matrix = np.array([[-x, -y, -1,0,0,0, x*x_prime, y*x_prime, x_prime],
                           [0,0,0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]])
    
    # Stack whole of the matrix
    for i in range(1, N):
        x = X[i][0]
        y = X[i][1]
        x_prime = Y[i][0]
        y_prime = Y[i][1]
        A_matrix_stack = np.array([[-x, -y, -1,0,0,0, x*x_prime, y*x_prime, x_prime],
                           [0,0,0, -x, -y, -1, x*y_prime, y*y_prime, y_prime]])
        A_matrix = np.vstack([A_matrix, A_matrix_stack])
    
    # Using SVD algorihtm to decompose
    u, sigma, vt = np.linalg.svd(A_matrix, full_matrices=False)
    # Get the smallest value column so, choose last row of the results
    homo_matrix = vt[-1].reshape(3,3)
    # Scale down
    homo_matrix = (homo_matrix / homo_matrix[2][2])
    H = homo_matrix
    
    ### END YOUR CODE   (do not modify this line)
    return H


def RANSAC_fit_homography(X, Y, size, threshold, nIters=1000):
    '''
    Perform RANSAC to find the homography with the most inliers

    Inputs:
        X: A numpy array of shape (N,2), each row is in the form [x_i,y_i]
        Y: A numpy array of shape (N,2), each row is in the form [x'_i,y'_i]
        size: Subset size for each iteration.
        threshold: Threshold for inlier calculation
        nIters: Number of iterations for running RANSAC

    Outputs:
        bestRefit: A homography matrix of shape (3,3) that satisfies H[-1,-1]=1
        fitting best to the inliers.

    Hint: np.random.choice(..., replace=False)
          For precise tuning, do not square the error when calculating inliers.
          Don't forget to **re-fit** the homography after iterations
    '''
    bestH, bestCount = np.eye(3), -1
    bestInliers = np.zeros((X.shape[0],), dtype=bool)
    bestRefit = np.eye(3)
    ### YOUR CODE HERE  (do not modify this line)
    
    N = X.shape[0]
    best_msa = 10e9
    for i in range(nIters):
        idx = np.random.choice(N, size, replace = False)
        cur_H = fit_homography(X[idx,:], Y[idx,:])
        X_trans = transform_homography(X, cur_H)
        
        msa = np.sum(np.abs(Y-X_trans), axis = 1)
        inliers = (msa < threshold)
        cur_count = np.count_nonzero(inliers)  
        
        if cur_count > bestCount:
            bestInliers = inliers
            bestH = cur_H
            bestCount = cur_count
        
    bestRefit = fit_homography(X[bestInliers], Y[bestInliers])
    
    ### END YOUR CODE   (do not modify this line)
    Y_in = Y[bestInliers]
    Y_in_pred_H = transform_homography(X[bestInliers], bestH)
    Y_in_pred_Refit = transform_homography(X[bestInliers], bestRefit)
    print(f'num_inliers / total: {bestCount}/{X.shape[0]}')
    print('err before re-fit:', 
          np.linalg.norm(Y_in_pred_H - Y_in, axis=1).mean())
    print('err after  re-fit:',
          np.linalg.norm(Y_in_pred_Refit - Y_in, axis=1).mean())
    return bestRefit


def scatter_plot(data, filename=None, s=.1):
    '''
    Draw scatter plots for Q1
    '''
    colors = ['b', 'r', 'g']
    for i in range(data.shape[1] // 2):
        plt.scatter(data[:,2*i], data[:,2*i+1], s=s, color=colors[i % 3])
    plt.axis([-1., 1.5, -1.5, 1.5])
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":
    os.makedirs('q1_result', exist_ok=True)

    # 1.2. Affine Transformation
    for num in [1, 2, 3]:
        XY = np.load(f'data/points_case_{num}.npy')
        M, t, Y_pred = fit_affine(XY[:,:2], XY[:,2:])
        print(f'points_case_{num} M matrix:\n', M,'\nt vector: ', t)
        scatter_plot(np.hstack([XY, Y_pred]), f'q1_result/2_affine_{num}.png')

    # 1.3. Homography
    for num in [1, 2, 3]:
        XY = np.load(f'data/points_case_{num}.npy')
        H = fit_homography(XY[:,:2], XY[:,2:])
        Y_pred = transform_homography(XY[:,:2], H)
        print(f'points_case_{num} H matrix:\n', H)
        scatter_plot(np.hstack([XY, Y_pred]), f'q1_result/3_homography_{num}.png')

    # 1.4. Homography via RANSAC
    ### YOUR CODE HERE  (do not modify this line)
    
    ransac_size = 5
    ransac_threshold = 0.15

    
    ### END YOUR CODE   (do not modify this line)
    for num in [1, 2, 3]:
        XY = np.load(f'data/points_case_{num}.npy')
        H = RANSAC_fit_homography(XY[:,:2], XY[:,2:],
                                  size=ransac_size, threshold=ransac_threshold)
        Y_pred = transform_homography(XY[:,:2], H)
        print(f'points_case_{num} H matrix:\n', H)
        scatter_plot(np.hstack([XY, Y_pred]), f'q1_result/4_ransac_{num}.png')

