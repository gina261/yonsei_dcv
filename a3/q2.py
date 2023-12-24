import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from common import read_img, save_img


def corner_score(image, offset=(5, 5), window_size=(5, 5)):
    """
    Given an input image, (x, y) offset, and window_size,
    compute the corner score E(u, v) with window W for all pixels.
    Use zero padding to handle window values outside of the image.

    Inputs:
    - image: A numpy array of shape (H, W)
    - offset: A tuple of (x, y) offsets = (u, v)
    - window_size: A tuple of window size = (kh, kw)

    Returns:
    - output: A numpy array of shape (H, W)

    Hint: The score function can be formulated as convolution, and it can be
          implemented in one ugly line of code.
          For shifting an image, `np.roll`
          For convolution, `scipy.ndimage.convolve()` with mode='constant'
    """
    H, W = image.shape
    u, v = offset
    kh, kw = window_size
    assert (kh % 2 == 1) and (kw % 2 == 1), 'Consider odd-sized windows only'
    output = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Roll the image    
    shifted_v_image = np.roll(image, v, axis=0)
    shifted_image = np.roll(shifted_v_image, u, axis = 1)
    
    # Using convolution to without using for-loop
    one_filter = np.ones((window_size[0], window_size[1]))    
    energy_function = ((image - shifted_image) ** 2) 
    output = scipy.ndimage.convolve(energy_function, one_filter, mode= 'constant')
    
    ### END YOUR CODE   (do not modify this line)
    return output


def harris_detector(image, window_size=(3, 3), alpha=0.05):
    """
    Given an input image, compute the Harris Detector score for all pixels.
    Use zero padding for any convolution/filtering to keep the original size.
    When computing M, use Gaussian weighting with sigma = .85,
    where the values are given for you.

    Inputs:
    - image: A numpy array of shape (H, W)
    - window_size: A tuple of window size
    - alpha: The constant adjusting the sensitivity of the detector

    Returns:
    - output: A numpy array of shape (H, W)

    Hint: For derivative, `scipy.ndimage.sobel()` with mode='constant'
          For convolution, `scipy.ndimage.convolve()` with mode='constant'
          Compute {Ix and Iy}, {Ixx, Ixy, Iyy}, {M}, output in sequence.
    """
    # Normalized Gaussian filter with sigma = .85
    gaussian = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]]) / 16

    H, W = image.shape
    kh, kw = window_size
    assert (kh % 2 == 1) and (kw % 2 == 1), 'Consider odd-sized windows only'
    output = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Y - axis H [0]
    # X - axis W [1]
    Iy = scipy.ndimage.sobel(image.copy(), axis = 0, mode = 'constant')
    Ix = scipy.ndimage.sobel(image.copy(), axis = 1,mode = 'constant')
    
    # Check image that come from previous 
    # save_img(Ix, f'tmp.png')
    # save_img(Iy,f'tmp2.png')
    
    Iyy = Iy ** 2
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyx = Iy * Ix
    
    a = scipy.ndimage.convolve(Ixx, gaussian, mode = 'constant') # M00
    b = scipy.ndimage.convolve(Ixy, gaussian, mode = 'constant') # M01
    c = scipy.ndimage.convolve(Ixy, gaussian, mode = 'constant') # M10
    d = scipy.ndimage.convolve(Iyy, gaussian, mode = 'constant') # M11
    
    M = np.array([[a, b],
                  [c, d]])
    
    # print(a.shape)
    # print(b.shape)
    # print(c.shape)    
    # print(d.shape)
    
    # # print(M)
    # print(b)
    # print(c)
    # print(M.shape)
    # Determinant ad - bc 
    D = ((a*d) - (b*c))
    
    # Trace Diagonal matrix
    trace = a + d
        
    R = D - alpha * (trace ** 2) 
    output = R
    
    ### END YOUR CODE   (do not modify this line)
    return output


def main():
    img = read_img('data/grace_hopper_noisy.png')

    os.makedirs('q2_result', exist_ok=True)

    # 2.1. Corner Score
    # (a) Complete corner_score(); compute the corner score map and their min
    W = (5,5)
    offsets = [(0,5), (0,-5), (5,0), (-5,0)]
    directions = ['b', 't', 'r', 'l']
    score_maps = []
    for offset, direction in zip(offsets, directions):
        score_map = corner_score(img, offset, W)
        save_img(score_map, f'q2_result/1_a_corner_score_{direction}.png')
        score_maps.append(score_map)
    min_score_map = np.stack(score_maps, axis=-1).min(axis=-1)
    save_img(min_score_map, f'q2_result/1_a_corner_score.png')

    # (b) No Code

    print('Corner Score done.')


    # 2.2. Harris Corner Detector
    # (a) Complete harris_detector()
    harris_corners = harris_detector(img)
    save_img(harris_corners, 'q2_result/2_a_harris_response.png')

    # (b) Tune `threshold` to get a good detection result
    threshold = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # for th in range(0,30):
    #     threshold = th * 0.1
    #         # heatmap to figure out the threshold
    #     plt.figure(figsize=(12,6))
    #     plt.pcolor(harris_corners[::-1])
    #     plt.colorbar()
    #     plt.savefig('q2_result/2_b_heatmap.png')
    #     plt.close()

    #     # 7x7 circular kernel for better corner detection visibility
    #     radius = 3
    #     kernel = np.zeros([radius*2+1]*2)
    #     for y in np.arange(-radius, radius+1):
    #         for x in np.arange(-radius, radius+1):
    #             if x**2 + y**2 <= radius**2:
    #                 kernel[y+radius,x+radius] = 1

    #     # highlight corners in the original image
    #     mask = harris_corners.copy()
    #     mask = scipy.ndimage.convolve(mask, kernel, mode='constant')
    #     mask[mask < threshold] = 0.
    #     mask[mask > 0] = 1.
    #     highlighted = np.stack([img+mask, img, img], axis=-1)
    #     save_img(highlighted, f'q2_result/2_b_harris_highlighted_{th}.png',
    #             normalize=False)

    #     print(f'Harris Corner Detector {th}done.')
    
    threshold = 2.5
    
    ### END YOUR CODE   (do not modify this line)

    # heatmap to figure out the threshold
    plt.figure(figsize=(12,6))
    plt.pcolor(harris_corners[::-1])
    plt.colorbar()
    plt.savefig('q2_result/2_b_heatmap.png')
    plt.close()

    # 7x7 circular kernel for better corner detection visibility
    radius = 3
    kernel = np.zeros([radius*2+1]*2)
    for y in np.arange(-radius, radius+1):
        for x in np.arange(-radius, radius+1):
            if x**2 + y**2 <= radius**2:
                kernel[y+radius,x+radius] = 1

    # highlight corners in the original image
    mask = harris_corners.copy()
    mask = scipy.ndimage.convolve(mask, kernel, mode='constant')
    mask[mask < threshold] = 0.
    mask[mask > 0] = 1.
    highlighted = np.stack([img+mask, img, img], axis=-1)
    save_img(highlighted, 'q2_result/2_b_harris_highlighted.png',
             normalize=False)

    print('Harris Corner Detector done.')


if __name__ == "__main__":
    main()
