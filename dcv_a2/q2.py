import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt


def split_triptych(trip):
    """
    Split a triptych into three channels and return a tuple of them

    Input:  trip: a triptych (H x W matrix)
    Output: a tuple of R, G, B matrices
    
    Note: if H % 3 != 0, you can remove the last few rows.
    """
    R, G, B = None, None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    vertical_line = trip.shape[0]
    
    # Make H to multiplier of 3 
    # INDEX should be integer
    interval = int((vertical_line - (vertical_line % 3)) / 3)
    B = trip[ :interval, :]
    G = trip[interval : 2*interval, :]
    R = trip[2*interval:3*interval, :]

    ### END YOUR CODE   (do not modify this line)
    return R, G, B


def ncc(ch1, ch2):
    """
    Calculates normalized cross correlation between 2 color channels

    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    
    Note: whether the output is normalized (i.e., the range of the pixels is
          either [0,1] or [0,255]) should not affect the output.

    Hint: np.linalg.norm
    """
    out = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Change with float values due to prevent overflow
    ch1 = ch1.copy().astype('float')
    ch2 = ch2.copy().astype('float')
    
    # Default L2 Norm, make to vector
    ch1_norm = np.linalg.norm(ch1.reshape(-1))
    ch2_norm = np.linalg.norm(ch2.reshape(-1))
    
    ncc_val = (np.dot(ch1.reshape(-1) , ch2.reshape(-1))) / (ch1_norm * ch2_norm) 
    out = ncc_val
    
    ### END YOUR CODE   (do not modify this line)
    return out


def best_offset(ch1, ch2, metric, Xrange, Yrange):
    """
    Use metric to align ch2 to ch1 and return optimal offset

    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in horizontal direction
            Yrange: range to search for optimal offset in vertical direction
    Output: optimal offset for (y,x)

    Note: Searching in Xrange would mean moving in the horizontal axis of
          the image/matrix, Yrange is the vertical axis 

    Hint: np.roll
    """
    offset = np.zeros(2, dtype=int)
    ### YOUR CODE HERE  (do not modify this line)
    
    # Make the whole grid cases for all of the pairwise coordinates
    X_grid , Y_grid = np.meshgrid(Xrange, Yrange)
    
    best_ncc_val = -1
    Xrange_arr  = np.arange(-Xrange, Xrange+1, 1)
    Yrange_arr  = np.arange(-Yrange, Yrange+1, 1)
    
    for x_idx in Xrange_arr:
        X_shifted_img = np.roll(ch2.copy(), x_idx, axis = 1)
        for y_idx in Yrange_arr:
            shifted_img = np.roll(X_shifted_img, y_idx, axis = 0)
            
            # If we roll all of the edges will be moved
            # We should compare what doens't affect with rolling
            # We will compare center Area which do not affect along
            cur_ncc = metric(ch1[Xrange:-Xrange, Yrange:-Yrange], shifted_img[Xrange:-Xrange, Yrange:-Yrange])
            # print(cur_ncc)
            if cur_ncc > best_ncc_val:
                
                # Update metric values
                # Highes ncc values will be same channels
                best_ncc_val = cur_ncc
                offset[0] = y_idx
                offset[1] = x_idx
     
    ### END YOUR CODE   (do not modify this line)
    return offset


def align_and_combine(R, G, B, metric, max_offset=20):
    """
    Use metric to align the three channels 

    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
            max_offset: bound of range for searching offsets
    Output: aligned RGB image 

    Hint: np.arange; Use R as the anchor to align G and B
    """
    image = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Fix R channel and Move G channel
    g_offset = best_offset(R, G, metric, max_offset, max_offset)
    
    # Offset Y_idx, X_idx  y_idx -> axis 0 , x_idx -> axis 1
    g_shifted = np.roll(G, g_offset[0], axis = 0)
    g_shifted = np.roll(g_shifted, g_offset[1], axis = 1)
    print(f"G Channel: {g_offset}")
    
    b_offset = best_offset(R, B, metric, max_offset, max_offset)
    
    # Offset Y_idx, X_idx  y_idx -> axis 0 , x_idx -> axis 1
    b_shifted = np.roll(B, b_offset[0], axis = 0)
    b_shifted = np.roll(b_shifted, b_offset[1], axis = 1)
    print(f"B Channel: {b_offset}")
    
    image = np.stack([R,g_shifted,b_shifted], axis = 2)
    
    # Crop Image for unaffected area
    image = image[max_offset:-max_offset, max_offset:-max_offset]
    
    ### END YOUR CODE   (do not modify this line)
    return image


def pyramid_align(R, G, B, metric, max_offset=10):
    """
    Perform the image pyramid alignment iteratively

    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
            max_offset: bound of range for searching offsets
    Output: aligned RGB image 

    Hint: cv2.resize
    """
    image = None
    ### YOUR CODE HERE  (do not modify this line)
    
    # Step 0: Down sample to max level
    origH , origW = R.shape[0], R.shape[1]
    
    lev = 2
    
    # Initialize total offset
    tot_G_offset = np.array([0,0])
    tot_B_offset = np.array([0,0])
    
    for lev in range(2,-1,-1):
        
        # Steo 1 : Down Sampled Image 
        curR = cv2.resize(R, (origW // (4 ** lev), (origH // (4 ** lev))))
        curG = cv2.resize(G, (origW // (4 ** lev), (origH // (4 ** lev)))) 
        curB = cv2.resize(B, (origW // (4 ** lev), (origH // (4 ** lev)))) 
        # Step 2 : Previous offset was affect by previous offset
        # Start to find aligned and new offset for current level
        print(f"-" * 50 )        
        print(f"Level {lev} : Current Image Size {curR.shape[0]} X {curR.shape[1]}")
        print(" Fidning Alignment offset ")
        
        tot_G_offset *= 4 
        tot_B_offset *= 4
         
        print(f"Before Level {lev} Offset : G : {tot_G_offset} B: {tot_B_offset}")
        
        g_shifted = np.roll(curG , tot_G_offset[0], axis = 0)
        g_shifted = np.roll(g_shifted, tot_G_offset[1], axis = 1)
        
        b_shifted = np.roll(curB, tot_B_offset[0], axis = 0)
        b_shifted = np.roll(b_shifted, tot_B_offset[1], axis = 1)
        
        G_offset = best_offset(curR, g_shifted, metric, max_offset, max_offset)
        B_offset = best_offset(curR, b_shifted, metric, max_offset, max_offset)
        
        print(f"Level {lev} Offset : G : {G_offset} B: {B_offset}")
        tot_G_offset += G_offset
        tot_B_offset += B_offset
        
        print(f"After Level {lev} Offset : G : {tot_G_offset} B: {tot_B_offset}")
        
    # Stacking whole of the images 
    
    final_G = np.roll(G, tot_G_offset[0], axis = 0)
    final_G = np.roll(final_G, tot_G_offset[1], axis = 1) 
    final_B = np.roll(B, tot_B_offset[0], axis = 0)
    final_B = np.roll(final_B, tot_B_offset[1], axis = 1) 
    
    image = np.stack([R, final_G, final_B], axis = 2)
    
    # Crop Image for unaffected area
    image = image[max_offset:-max_offset, max_offset:-max_offset]
    
    ### END YOUR CODE   (do not modify this line)
    return image


def main():
    os.makedirs('q2_result', exist_ok=True)

    image = plt.imread('images/emir.jpg')
    print('Q2(a)')
    # (a) Generate a color image by splitting the triptych image
    # Hint: np.stack
    ### YOUR CODE HERE  (do not modify this line)
    
    curR, curG, curB = split_triptych(image)
    sum_img = np.stack([curR, curG, curB], axis = 2)
    plt.imsave('./q2_result/emir_color.jpg', sum_img)
    
    ### END YOUR CODE   (do not modify this line)

    print('Q2(b)')
    # (b) Remove misalignment in the color channels by calculating best offset
    ### YOUR CODE HERE  (do not modify this line)
    
    # Test for align_and_combine Function
    aligned_img = align_and_combine(curR, curG, curB, ncc)
    plt.imsave('./q2_result/emir_align.jpg', aligned_img)
    
    ### END YOUR CODE   (do not modify this line)

    print('Q2(c)')
    for image_name in ['seoul', 'vancouver']:
        image = plt.imread(f'images/{image_name}.jpg')
        # (c) Pyramid alignment
        ### YOUR CODE HERE  (do not modify this line)
        
        curR, curG, curB = split_triptych(image)
        aligned_img = pyramid_align(curR, curG, curB, ncc)
        plt.imsave(f'./q2_result/{image_name}_align.jpg', aligned_img)
        
        ### END YOUR CODE   (do not modify this line)


if __name__ == "__main__":
    main()
