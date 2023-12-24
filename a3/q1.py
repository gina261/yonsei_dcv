import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from common import read_img, save_img


def convolve(image, kernel):
    """
    Perform 2D convolution: image * kernel.
    Note that you should implement "convolution", not "cross-correlation".
    For consistency purposes, please use zero padding. During padding, zeros
    should be placed symmetrically (i.e., equally on both sides) along
    the height and width axes of the input. Be careful not to modify
    the input image directly.

    Inputs:
    - image: A numpy array of shape (H, W)
    - kernel: A numpy array of shape (kh, kw)

    Returns:
    - output: A numpy array of shape (H, W)

    Hint: For zero padding, `np.pad` with the default mode: mode='constant'
          For flipping the kernel, `np.flip`
    """
    H, W = image.shape
    kh, kw = kernel.shape
    assert (kh % 2 == 1) and (kw % 2 == 1), 'Consider odd-sized kernels only'
    output = None
    
    
    ### YOUR CODE HERE
    # flipped_filter = kernel.copy()
    flipped_filter = np.flip(kernel.copy())
    
    # Pad width ((Top, bottom), (Left, Right))
    padded_input = np.pad(image.copy(), mode= 'constant', pad_width = ((kh//2, kh//2),(kw//2,kw//2)), constant_values=0)
    output = np.zeros((H,W))
    
    for curh in range(H):
        for curw in range(W):
            conv_val = 0.0
            # Convolution curh, curw with same place
            cur_looking_mat = padded_input[curh:curh+kh, curw:curw+kw]
            conv_val = np.sum(cur_looking_mat * flipped_filter)
            output[curh, curw] = conv_val
    ### END YOUR CODE
    
    
    return output


def gaussian_1d(sigma, size, normalize=False, laplacian=False):
    """
    Compute 1D Gaussian kernel weights.

    Inputs:
    - sigma: The standard deviation of the Gaussian kernel.
    - size: The size of the Gaussian kernel.
    - normalize: If True, normalize weights s.t. they sum up to 1.
    - laplacian: If True, compute LoG instead.

    Returns:
    - kernel: A numpy array of shape (size,)

    Hint: `np.arange` might be useful.
          LoG = (x^2 / sigma^2 - 1) * Gaussian
    """
    assert size % 2 == 1, 'Consider odd-sized kernels only'
    kernel = None
    
    
    ### YOUR CODE HERE
    # Should be center is 0 values for relative values
    base_x = np.arange(-(size//2),(size//2)+1)
    exp_val = np.exp((-(base_x ** 2)) / (2* (sigma **2))) * (1/np.sqrt(2*np.pi*(sigma ** 2)))
    
    if normalize:
        kernel = exp_val / np.sum(exp_val)
    else:
        kernel = exp_val
    
    if laplacian:
        Log_kernel = exp_val * (((base_x ** 2)/ ((sigma ** 2))) -1)
        kernel = Log_kernel
        
    # print(kernel)
    # print(f'Sum of the 1d kernel: {np.sum(kernel)}')
    ### END YOUR CODE
    
    
    return kernel


def gaussian_2d(sigma, size, normalize=False, laplacian=False):
    """
    Compute 2D Gaussian kernel weights.

    Inputs:
    - sigma: The standard deviation of the Gaussian kernel.
    - size: The size of the Gaussian kernel.
    - normalize: If True, normalize weights s.t. they sum up to 1.
    - laplacian: If True, compute LoG instead.

    Returns:
    - kernel: A numpy array of shape (size,size)

    Hint: `gaussian_1d()` might be useful.
    """
    assert size % 2 == 1, 'Consider odd-sized kernels only'
    kernel = None
    ### YOUR CODE HERE
    
    if normalize:
        kernel_x = gaussian_1d(sigma, size, normalize = True, laplacian= False)
    else:    
        kernel_x = gaussian_1d(sigma, size, normalize = False, laplacian= False)
    
    kernel_2d = kernel_x.reshape(-1,1)
    kernel = kernel_2d @ kernel_2d.T
    
    # Normalize that sum of the value is 1
    if normalize:
        kernel = kernel / np.sum(kernel)
    
    if laplacian:
        kernel_x = gaussian_1d(sigma, size, normalize=False, laplacian=True)
        l1 = np.tile(kernel_x, (kernel_x.shape[0],1))
        l2 = np.tile(kernel_x.reshape(-1,1), (1, kernel_x.shape[0]))
        kernel = (l1 + l2)
    
    # print(f'Sum of the 2d Kernel {np.sum(kernel)}')
    # print(kernel)

    ### END YOUR CODE
    return kernel


def detect_edge(image, sobel=False):
    """
    Compute Ix, Iy, and the gradient magnitude of the input image.

    Inputs:
    - image: A numpy array of shape (H, W)
    - sobel: If True, apply 3x3 sobel filters.
             If False, apply 1x3 and 3x1 derivative filters.

    Returns a tuple of:
    - Ix: A numpy array of shape (H, W)
    - Iy: A numpy array of shape (H, W)
    - grad_magnitude: A numpy array of shape (H, W)

    Hint: Use your `convolve` for convolution.
          Be aware of the direction of the grad; `convolve` flips the kernel.
    """
    Ix, Iy, grad_magnitude = None, None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    diff_x = np.array([1,0,-1]).reshape(1,-1)
    diff_y = np.array([1,0,-1]).reshape(-1,1)
    
    
    #Sobel Mode
    if sobel:
        sobel_x = np.array([[1,0,-1],
                            [2,0,-2],
                            [1,0,-1]])
        sobel_y = np.array([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]])
        
        Ix = convolve(image.copy(), sobel_x)
        Iy = convolve(image.copy(), sobel_y)
        
    # Generalized Mode
    else:
        Ix = convolve(image.copy(), diff_x)
        Iy = convolve(image.copy(), diff_y)
    grad_magnitude = np.sqrt(Ix ** 2  + Iy ** 2)
    
    ### END YOUR CODE   (do not modify this line)
    return Ix, Iy, grad_magnitude


def main():
    img = read_img('data/grace_hopper_noisy.png')

    os.makedirs('q1_result', exist_ok=True)

    # 1.1. Convolution and Gaussian Filter
    # (a) No code

    # (b) Complete convolve(); diff should be small enough, < 1e-8
    kernel = np.random.randn(5,3)
    filtered = convolve(img, kernel)
    expected = scipy.ndimage.convolve(img, kernel, mode='constant')
    diff = np.linalg.norm(filtered - expected)
    print('convolve diff from scipy:', diff)

    # (c) Complete gaussian_1d(*args, laplacian=False) and generate images.
    sigma, ksize = 1, 3
    for normalize in [False, True]:
        kernel_gaussian_2d = gaussian_2d(sigma, ksize, normalize=normalize)
        filtered_gaussian = convolve(img, kernel_gaussian_2d)

        if normalize:
            save_str = 'q1_result/1_c_gaussian_normalized.png'
        else:
            save_str = 'q1_result/1_c_gaussian_unnormalized.png'
        save_img(filtered_gaussian, save_str, normalize=False)

    # (d): No code

    # (e): Complete detect_edge(*args, sobel=False) and generate images.
    Ix_naive, Iy_naive, edge_naive = detect_edge(img)
    save_img(Ix_naive, f'q1_result/1_e_ix.png')
    save_img(Iy_naive, f'q1_result/1_e_iy.png')
    save_img(edge_naive, f'q1_result/1_e_edge.png')
    Ix_gaussian, Iy_gaussian, edge_gaussian = detect_edge(filtered_gaussian)
    save_img(Ix_gaussian, f'q1_result/1_e_ix_gaussian.png')
    save_img(Iy_gaussian, f'q1_result/1_e_iy_gaussian.png')
    save_img(edge_gaussian, f'q1_result/1_e_edge_gaussian.png')

    print('Convolution and Gaussian Filter done.')


    # 1.2. Sobel Operator
    # (a): No code

    # (b): Complete detect_edge(*args, sobel=True) and generate images.
    Ix_sobel, Iy_sobel, edge_sobel = detect_edge(img, sobel=True)
    save_img(Ix_sobel, f'q1_result/2_b_ix_sobel.png')
    save_img(Iy_sobel, f'q1_result/2_b_iy_sobel.png')
    save_img(edge_sobel, f'q1_result/2_b_edge_sobel.png')

    print('Sobel Operator is done.')


    # 1.3. LoG Filter
    # (a) Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    kernel_LoG1 = np.array([[0,  1, 0],
                            [1, -4, 1],
                            [0,  1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])

    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    save_img(filtered_LoG1, 'q1_result/3_a_LoG1.png')
    save_img(filtered_LoG2, 'q1_result/3_a_LoG2.png')

    # (b): No code

    # (c) Complete gaussian_1d(*args, laplacian=True) and compare LoG and DoG.
    sigma, ratio, ksize = 50, 1.06, 501
    LoG = gaussian_1d(sigma, ksize, laplacian=True) * (ratio - 1)
    DoG = gaussian_1d(sigma*ratio, ksize) - gaussian_1d(sigma, ksize)
    plt.figure()
    plt.plot(LoG, 'r-')
    plt.plot(DoG, 'b-')
    plt.savefig('q1_result/3_c_LoG_vs_DoG.png')
    plt.close()

    print('LoG Filter is done.')


if __name__ == "__main__":
    main()
