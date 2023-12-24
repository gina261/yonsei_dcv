import os
import numpy as np
import scipy.ndimage
from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)
from q1 import gaussian_2d


def gaussian_filter(image, sigma):
    """
    Given an input image, apply a Gaussian filter.
    Use reflect padding.

    Inputs:
    - image: A numpy array of shape (H, W)
    - sigma: The standard deviation of the Gaussian kernel.

    Returns:
    - output: A numpy array of shape (H, W)

    Hint: For convolution, `scipy.ndimage.convolve()` with mode='reflect'
          `gaussian_2d()` you implemented in Q1 might be useful.
          We already provide you a heuristic way of setting kernel size,
          so simply apply Gaussian filtering of size kernel_size x kernel_size.
    """
    H, W = image.shape
    # a heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    # ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    output = None
    ### YOUR CODE HERE  (do not modify this line)
    
    cur_filter = gaussian_2d(sigma, kernel_size, normalize = True)
    # print(cur_filter.shape)
    result = scipy.ndimage.convolve(image.copy(), cur_filter , mode = 'reflect')
    output = result
    
    ### END YOUR CODE   (do not modify this line)
    return output


def main():
    image = read_img('data/polka.png')

    os.makedirs('q3_result', exist_ok=True)

    # 3.1. Single-scale Blob Detection
    # (a, b) Complete gaussian_filter(); detecting polka dots

    # detect small circles
    sigma_1, sigma_2 = None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    sigma_1 = 3.01
    # LOG filter 1-2
    sigma_2 = sigma_1 * 1.6 
    
    ### END YOUR CODE   (do not modify this line)
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)
    DoG_small = gauss_2 - gauss_1

    # visualize maxima
    # value of k_xy is a suggestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_small, k_xy=10)
    print('Number of small circles: ', len(maxima))
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          'q3_result/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     'q3_result/polka_small.png')

    # detect large circles
    sigma_1, sigma_2 = None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    sigma_1 = 11
    sigma_2= sigma_1 * 1.6
    
    ### END YOUR CODE   (do not modify this line)
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)
    DoG_large = gauss_2 - gauss_1

    # visualize maxima
    # value of k_xy is a suggestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    print('Number of large circles: ', len(maxima))
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          'q3_result/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     'q3_result/polka_large.png')

    print('Single-scale Blob Detection done.')


    # 3.2. Cell Counting
    for filename in ['001cell', '002cell', '004cell', '005cell']:
        image = read_img(f'data/{filename}.png')
        sigma_1, sigma_2 = None, None
        ### YOUR CODE HERE  (do not modify this line)
        
        small_scale_sigma1 = 1.0
        small_scale_sigma2 = small_scale_sigma1 * 1.414
        small_dog = gaussian_filter(image.copy(), small_scale_sigma2) - gaussian_filter(image.copy(), small_scale_sigma1)
        
        mid_scale_sigma1 = 3.0
        mid_scale_sigma2 = mid_scale_sigma1 * 1.414
        mid_dog = gaussian_filter(image.copy(), mid_scale_sigma2) - gaussian_filter(image.copy(), mid_scale_sigma1)
        
        large_scale_sigma1 = 5.0
        large_scale_sigma2 = large_scale_sigma1 * 1.414
        large_dog = gaussian_filter(image.copy(), large_scale_sigma2) - gaussian_filter(image.copy(), large_scale_sigma1)
        
        gigantic_scale_sigma1 = 7.0
        gigantic_scale_sigma2 = gigantic_scale_sigma1 * 1.414
        gigantic_dog = gaussian_filter(image.copy(), gigantic_scale_sigma2) - gaussian_filter(image.copy(), gigantic_scale_sigma1)
        
        total_img = np.stack([small_dog, mid_dog, large_dog, gigantic_dog], axis = 2)
           
        total_cnt = find_maxima(total_img, k_xy = 11, k_s = 4)
                
        # visualize_scale_space(large_dog, large_scale_sigma1, large_scale_sigma2 / large_scale_sigma1,
        #                   f'q3_result/{filename}_large.png')
        # visualize_scale_space(mid_dog, mid_scale_sigma1, mid_scale_sigma2 / mid_scale_sigma1,
        #                   f'q3_result/{filename}_mid.png')
        visualize_scale_space(small_dog, small_scale_sigma1, small_scale_sigma2 / small_scale_sigma1,
                          f'q3_result/{filename}_small_large.png')
        visualize_maxima(image, total_cnt, small_scale_sigma1, small_scale_sigma2 / small_scale_sigma1,
                        f'q3_result/{filename}.png')
        
        print(f'Total Cells {filename} : {len(total_cnt)}')
        
        ### END YOUR CODE   (do not modify this line)
    print('Cell Counting done.')


if __name__ == '__main__':
    main()
