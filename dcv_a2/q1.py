import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils import generate_gif, renderCube


def rotX(theta):
    """
    Generate 3D rotation matrix about x-axis
    Input:  theta: rotation angle along x-axis
    Output: Rotation matrix (3 x 3 array)
    """
    mat = None
    ### YOUR CODE HERE  (do not modify this line)
    
    mat = np.array([[1, 0, 0], 
                    [0, np.cos(theta),-np.sin(theta)], 
                    [0, np.sin(theta), np.cos(theta)]])
    
    ### END YOUR CODE   (do not modify this line)
    return mat


def rotY(theta):
    """
    Generate 3D rotation matrix about y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    mat = None
    ### YOUR CODE HERE  (do not modify this line)
    
    mat = np.array([[np.cos(theta), 0, np.sin(theta)], 
                    [0, 1, 0], 
                    [-np.sin(theta), 0, np.cos(theta)]])
    
    ### END YOUR CODE   (do not modify this line)
    return mat


def main():
    os.makedirs('q1_result', exist_ok=True)

    print('Q1(a)')
    # (a) Use rotY() to generate cube.gif with 24 frames.
    # Hint: np.linspace(). endpoint=False makes the rotation smooth.
    ### YOUR CODE HERE  (do not modify this line)
    
    rotLi = np.linspace(0, 2*np.pi, 24, endpoint = False)
    tot_rotation = []
    for theta in rotLi:
        tot_rotation.append(rotY(theta))
    # print(tot_rotation)
    generate_gif(tot_rotation)
    
    ### END YOUR CODE   (do not modify this line)

    print('Q1(b)')
    # (b) Use rotX() and rotY() to check if rotation matrices are commutative.
    ### YOUR CODE HERE  (do not modify this line)
    
    os.makedirs('q1_result', exist_ok=True)
    
    rotxy = rotY((np.pi)/4) @ rotX((np.pi)/4) 
    rotyx = rotX((np.pi)/4) @ rotY((np.pi)/4)

    #xy
    xy_fig = renderCube(R=rotxy, t = np.array([0,0,3]))
    
    # Get from utils.generate_fig helper function
    xy_fig.canvas.draw()
    img_xy = np.frombuffer(xy_fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_xy = img_xy.reshape(xy_fig.canvas.get_width_height()[::-1] + (3, ))
    plt.imsave('q1_result/q1b_xy.png', img_xy)

    #yx
    yx_fig = renderCube(R=rotyx, t = np.array([0,0,3]))
    yx_fig.canvas.draw()
    img_yx = np.frombuffer(yx_fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_yx = img_yx.reshape(yx_fig.canvas.get_width_height()[::-1] + (3, ))
    plt.imsave('q1_result/q1b_yx.png', img_yx)
    
    ### END YOUR CODE   (do not modify this line)
    
    print('Q1(c)')
    # (c) Combine rotX() and rotY() to render a cube projection such that
    #     end points of diagonal overlap
    # Hint: Try with multiple configurations to narrow down the search region
    ### YOUR CODE HERE  (do not modify this line)
    
    # Arcsin (root(2/3))
    # Wrong Answer... Numerical Search
    # for i in range(600,700):
    #     tot_R = rotX(i * 0.001) @ rotY(np.pi/4)
    tot_R = rotX(np.pi/5) @ rotY(np.pi/4)
    tot_R = rotX(0.609) @ rotY(np.pi/4)
    
    
    diag_fig = renderCube(R=tot_R, t = np.array([0,0,3]))
    diag_fig.canvas.draw()
    img_diag = np.frombuffer(diag_fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_diag = img_diag.reshape(diag_fig.canvas.get_width_height()[::-1] + (3, ))
    
    plt.imsave('q1_result/q1c.png', img_diag)
    
    ### END YOUR CODE   (do not modify this line)


if __name__ == "__main__":
    main()
