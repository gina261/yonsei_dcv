import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def change_value(image):
    """
    Change the value of every pixel by following x_n = 0.5*x_p^2 
    where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape (image_height, image_width, 3)

    Returns:
        out: numpy array of shape (image_height, image_width, 3)
    """
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Change to floating value that input is uint 8 
    float_img = np.zeros((image.shape[0], image.shape[1], image.shape[2]), dtype = float)
    float_img = image / (255.0)
    float_img = (float_img ** 2) * (0.5)
    
    # Remake to uint8 format
    out = np.uint8( float_img * 255)
    
    ### END YOUR CODE   (do not modify this line)
    return out


def convert_to_gray(image):
    """
    Change image to gray scale

    Args:
        image: numpy array of shape (image_height, image_width, 3)

    Returns:
        out: numpy array of shape (image_height, image_width)
    
    Hint: cv2.cvtColor
    """ 
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    change_color = cv2.COLOR_BGR2GRAY
    
    cur_img = image.copy()
    gray_img = cv2.cvtColor(cur_img, change_color)
    out = gray_img
    
    ### END YOUR CODE   (do not modify this line)
    return out


def rgb_decomposition(image, channel):
    """
    Return image **excluding** one of the RGB channel specified

    Args:
        image: numpy array of shape (image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape (image_height, image_width, 3)
    """
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    cur_img = image.copy()
    change_color = cv2.COLOR_BGR2RGB
    recap_color = cv2.COLOR_RGB2BGR
    rgb_img = cv2.cvtColor(cur_img, change_color)
    
    if channel == 'R':
        rgb_img[:,:,0] = 0
    elif channel == 'G':
        rgb_img[:,:,1] = 0
    elif channel == 'B':
        rgb_img[:,:,2] = 0
    else:
        print('Error check Channel')
    
    out = cv2.cvtColor(rgb_img, recap_color)

    
    ### END YOUR CODE   (do not modify this line)
    return out


def hsv_decomposition(image, channel='H'):
    """
    Return image **excluding** one of the HSV channel specified

    Args:
        image: numpy array of shape (image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape (image_height, image_width, 3)

    Hint: make sure that `out` is a BGR image.
    """
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    cur_img = image.copy()
    change_color = cv2.COLOR_BGR2HSV
    recap_color = cv2.COLOR_HSV2BGR
    rgb_img = cv2.cvtColor(cur_img, change_color)
    
    if channel == 'H':
        rgb_img[:,:,0] = 0
    elif channel == 'S':
        rgb_img[:,:,1] = 0
    elif channel == 'V':
        rgb_img[:,:,2] = 0
    else:
        print('Error check Channel')
    
    out = cv2.cvtColor(rgb_img, recap_color)
    
    ### END YOUR CODE   (do not modify this line)
    return out


def lab_decomposition(image, channel='L'):
    """
    Return image **excluding** one of the LAB channel specified

    Args:
        image: numpy array of shape (image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape (image_height, image_width, 3) 

    Hint: make sure that `out` is a BGR image.
    """
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    change_color = cv2.COLOR_BGR2LAB
    recap_color = cv2.COLOR_LAB2BGR
    cur_img = image.copy()
    rgb_img = cv2.cvtColor(cur_img, change_color)
    
    if channel == 'L':
        rgb_img[:,:,0] = 0
    elif channel == 'A':
        rgb_img[:,:,1] = 0
    elif channel == 'B':
        rgb_img[:,:,2] = 0
    else:
        print('Error check Channel')
    
    out = cv2.cvtColor(rgb_img, recap_color)
    
    ### END YOUR CODE   (do not modify this line)
    return out


def main():
    os.makedirs('q3_result', exist_ok=True)
    image = cv2.imread('images/baboon.png')

    print('Q3(a)')
    # (a) Convert the image using the formula x_n = 0.5*x_p^2.
    image_new = change_value(image)
    cv2.imwrite('q3_result/q3a.png', image_new)

    print('Q3(b)')
    # (b) Convert the image into gray scale.
    image_gray = convert_to_gray(image)
    cv2.imwrite('q3_result/q3b.png', image_gray)

    print('Q3(c)')
    # (c) Decomposes the input image into RGB and return the image
    #     **excluding** the specified channel.
    image_r = rgb_decomposition(image, 'R')
    image_g = rgb_decomposition(image, 'G')
    image_b = rgb_decomposition(image, 'B')
    cv2.imwrite('q3_result/q3c_no_r.png', image_r)
    cv2.imwrite('q3_result/q3c_no_g.png', image_g)
    cv2.imwrite('q3_result/q3c_no_b.png', image_b)

    print('Q3(d)')
    # (d) Decomposes the input image into HSV and return the image
    #     **excluding** the specified channel.
    image_h = hsv_decomposition(image, 'H')
    image_s = hsv_decomposition(image, 'S')
    image_v = hsv_decomposition(image, 'V')
    cv2.imwrite('q3_result/q3d_no_h.png', image_h)
    cv2.imwrite('q3_result/q3d_no_s.png', image_s)
    cv2.imwrite('q3_result/q3d_no_v.png', image_v)

    print('Q3(e)')
    # (e) Decomposes the input image into LAB and return the image
    #     **excluding** the specified channel.
    image_l = lab_decomposition(image, 'L')
    image_a = lab_decomposition(image, 'A')
    image_b = lab_decomposition(image, 'B')
    cv2.imwrite('q3_result/q3e_no_l.png', image_l)
    cv2.imwrite('q3_result/q3e_no_a.png', image_a)
    cv2.imwrite('q3_result/q3e_no_b.png', image_b)


if __name__ == "__main__":
    main()

