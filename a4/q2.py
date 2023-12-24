import cv2
import os
import numpy as np
from common import save_img, read_img, get_match_points, get_AKAZE
from q1 import transform_homography, fit_homography, RANSAC_fit_homography


def compute_distance(desc1, desc2):
    '''
    Calculates **squared** L2 distance between 2 binary descriptor vectors.

    Inputs:
        desc1: Descriptor vector of shape (N,F)
        desc2: Descriptor vector of shape (M,F)

    Outputs:
        dist: L2 distance matrix of shape (N,M), where dist[i,j] is
        the **squared** Euclidean distance between desc1[i] and desc2[j].

    Hint: You should use the distance calculation trick
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    '''
    dist = None
    ### YOUR CODE HERE  (do not modify this line)
    
    X_diag = np.diag(desc1 @ desc1.T)
    Y_diag = np.diag(desc2 @ desc2.T)
    
    # desc1 (569, 488)
    # desc2 (784, 488)
    
    # dist shape (569, 784)
    
    dist = X_diag.reshape(-1, 1) + Y_diag.reshape(1, -1) - 2 * desc1@ desc2.T
    
    ### END YOUR CODE   (do not modify this line)
    return dist


def find_matches(desc1, desc2, match_ratio):
    '''
    Calculates the **bijective** matches between the two sets of keypoint
    descriptors based on distance and ratio test.

    Inputs:
        desc1: Descriptor vector of shape (N,F)
        desc2: Descriptor vector of shape (M,F)
        match_ratio: maximum acceptable distance ratio between
        2 nearest matches; should be less than 1

    Outputs:
        matches: a list of indices of shape (K,2), where each row contains
        [i,j] giving the matches between desc1[i] and desc2[j].

    Hint: np.argsort. Python native set might also be useful.
    '''
    matches = []
    ### YOUR CODE HERE  (do not modify this line)
    
    # Image1 to Image 2 mathcing
    
    coord1 = []
    
    # dist1_2 shape (569,784)
    # dist2_1 shape (784,569)
    dist1_2 = compute_distance(desc1, desc2)
    dist2_1 = compute_distance(desc2, desc1)
    
    for i in range(dist1_2.shape[0]):
        sorted_npy = np.argsort(dist1_2[i])
        n1 = sorted_npy[0]
        n2 = sorted_npy[1]
        if match_ratio > (dist1_2[i][n1] / dist1_2[i][n2]) :
            coord1.append([i, n1])
    
    coord2 = []
    
    # Image2 to Image 1 matching
    # Same Process
    for j in range(dist1_2.shape[1]):
        
        sorted_npy = np.argsort(dist2_1[j])
        n1 = sorted_npy[0]
        n2 = sorted_npy[1]
        
        if match_ratio > (dist2_1[j][n1] / dist2_1[j][n2]):
            coord2.append([n1, j])
    
    # Using set to get intersection for the same feature 
    # img 1 -> img 2 (coord1) img 2 -> img 1 (coord2), Intersection will be same
    match_set = set(list(map(tuple, coord1))) & set(list(map(tuple, coord2)))
    matches = np.array(list(match_set))
    
    ### END YOUR CODE   (do not modify this line)
    return matches


def draw_matches(img1, img2, match_points):
    '''
    Creates an output image where the two input images stacked vertically
    connecting matching keypoints with a line. 

    Inputs:
        img1: Input image 1 of shape (h,w,3)
        img2: Input image 2 of shape (h,w,3)
        match_points: A list of indices of shape (K,4), where each row contains
        [i,j,i',j'] implying the matches between img1[i,j] and img2[i',j']

    Outputs:
        out: Out image of shape (h,2w,3), where two input images stacked
        vertically connecting matching keypoints with a line. 
    '''
    def get_color():
        return (np.random.rand(3)* 255).astype(int).tolist()
    out = np.concatenate([img1, img2], axis=1) # assume H1==H2, W1==W2
    offset = img1.shape[1]
    for i, (x1, y1, x2, y2) in enumerate(match_points):
        color = get_color()
        cv2.circle(out, (int(x1),int(y1)),
                   radius=4, color=color, thickness=2)
        cv2.circle(out, (int(x2)+offset,int(y2)),
                   radius=4, color=color, thickness=2)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+offset,int(y2)),
                 color=color, thickness=2)
    return out


def get_output_space(img1, img2, H):
    """
    Inputs:
        img1: Reference image of shape (h1,w1,3)
        img2: Image to be transformed of shape (h2,w2,3)
        H: Homography matrix of shape (3,3) mapping points in img2 to img1

    Outputs:
        output_shape: New shape (w',h') where all images lie within the space
        offset: Offset added to all images to make them lie within output_shape
    """
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    output_shape, offset = np.array([0,0]), np.array([0,0])
    ### YOUR CODE HERE  (do not modify this line)
    
    # img1 corner
    corner_orig = np.array([[0,0],
                            [0,h1],
                            [w1,h1],
                            [w1,0]])

    # img2 corner
    corner = np.array([[0,0],
                        [0,h2],
                        [w2,h2],
                        [w2,0]])

    # Transformation corner img2 to H
    trans = transform_homography(corner, H)    
    
    print(trans)
        
    # get 4 points of minimum values
    bottom = np.min([corner_orig[:,0], trans[:,0]])
    left = np.min([corner_orig[:,1], trans[:,1]])
    top = np.max([corner_orig[:,0], trans[:,0]])
    right = np.max([corner_orig[:,1], trans[:,1]])
    
    corner_min = np.array([bottom, left])
    output_shape = np.array([top, right]) - corner_min
    
    ### END YOUR CODE   (do not modify this line)

    # Convert to integer
    output_shape = np.ceil(output_shape).astype(int)
    offset = np.ceil(corner_min).astype(int)
    print('output shape:', output_shape)
    print('offset:', offset)

    return output_shape, offset


def warp_images(img1, img2, match_points, ransac_size, ransac_threshold):
    '''
    Take two images and return an image, putting together the full pipeline.
    You should return an image of the panorama put together.

    Inputs:
        img1: Input image 1 of shape (h,w,3)
        img2: Input image 2 of shape (h,w,3)
        match_points: A list of indices of shape (K,4), where each row contains
        [i,j,i',j'] implying the matches between img1[i,j] and img2[i',j']
        ransac_size: Subset size for each iteration in RANSAC
        ransac_threshold: Threshold for inlier calculation for RANSAC

    Outputs:
        img1_warped: Output image 1 of shape (h',w',3) aligned with img2_warped
        img2_warped: Output image 2 of shape (h',w',3) aligned with img1_warped

    Hint: RANSAC_fit_homography, get_output_space, cv2.warpPerspective
          To make img1 as a reference, find a homography from img2 to img1.
    '''
    img1_warped, img2_warped = None, None
    ### YOUR CODE HERE  (do not modify this line)
    
    # get Homography to get img1 references
    img2_coords = match_points[:, 2:]
    img1_coords = match_points[:, :2]
    
    H2_1 = RANSAC_fit_homography(img2_coords, img1_coords, ransac_size, ransac_threshold)
    outputs_shape, offset = get_output_space(img1, img2, H2_1)
    
    # Transition to matrix
    trans_matrix = np.array([[1, 0, -offset[0]], 
                  [0, 1, -offset[1]], 
                  [0, 0, 1]], dtype = float)

    # Img 2 to 1 than transformation
    img2H = trans_matrix @ H2_1
    
    # Img1 only transition matirx
    img1_warped = cv2.warpPerspective(img1, trans_matrix, outputs_shape)
    # Img2 to Img1 Homography, trasition matrix
    img2_warped = cv2.warpPerspective(img2, img2H, outputs_shape)
    
    ### END YOUR CODE   (do not modify this line)
    return img1_warped, img2_warped


def overlap_images(img1, img2):
    """
    Inputs:
        img1: Input image 1 of shape (h,w,3)
        img2: Input image 2 of shape (h,w,3)

    Outputs:
        out: Stitched image of shape (h,w,3); overlapping img1 onto img2
    """
    return img1 + img2 * (np.linalg.norm(img1, axis=2) == 0)[:,:,None]

def merge_images(img1, img2):
    """
    Inputs:
        img1: Input image 1 of shape (h,w,3)
        img2: Input image 2 of shape (h,w,3)

    Outputs:
        out: Stitched image of shape (h,w,3); overlapped region is averaged

    Hint: the background generated by cv2.warpPerspective() is in black, i.e.,
          the norm of the pixel not touched by input images is 0. You can use
          this fact to generate a mask.
    """
    out = np.zeros_like(img1)
    ### YOUR CODE HERE  (do not modify this line)
    
    img1_region = img1.copy() > 0
    img2_region = img2.copy() > 0
    overlapped_region = img1_region & img2_region
    
    # Make img1 only region
    # Make img2 only region
    img1_region = img1_region & (~overlapped_region)
    img2_region = img2_region & (~overlapped_region)
    
    out = img1_region * img1 + (overlapped_region) * 0.5 * img1 + (overlapped_region) * 0.5 * img2 + img2_region * img2 
    
    ### END YOUR CODE   (do not modify this line)
    return out


def blend_images(img1, img2, **kwargs):
    """
    Inputs:
        img1: Input image 1 of shape (h,w,3)
        img2: Input image 2 of shape (h,w,3)

    Outputs:
        out: Stitched image of shape (h,w,3). If you implement alpha blending,
        the overlapped region is blended by the relative horizontal position
        (1-alpha)*img1 + alpha*img2 where 0 <= alpha <= 1. Be aware of
        the relative position between img1 and img2.

    Hint: the background generated by cv2.warpPerspective() is in black, i.e.,
          the norm of the pixel not touched by input images is 0. You can use
          this fact to generate a mask.
    """
    out = np.zeros_like(img1)
    ### YOUR CODE HERE  (do not modify this line)
    
    alpha = kwargs['alpha']
    img1_region = img1.copy() > 0
    img2_region = img2.copy() > 0
    overlapped_region = img1_region & img2_region
    
    # Make img1 only region
    # Make img2 only region
    img1_region = img1_region & (~overlapped_region)
    img2_region = img2_region & (~overlapped_region)
    
    out = img1_region * img1 + (overlapped_region) * alpha * img1 + (overlapped_region) * (1-alpha) * img2 + img2_region * img2
     
    ### END YOUR CODE   (do not modify this line)
    return out


if __name__ == "__main__":
    os.makedirs('q2_result', exist_ok=True)

    ### YOUR CODE HERE  (do not modify this line)
    match_ratio = 0.83
    ransac_size = 7
    ransac_threshold = 0.25
    ### END YOUR CODE   (do not modify this line)

    filename = 'diamond_head'
    img1 = read_img(f'data/{filename}1.jpg')
    img2 = read_img(f'data/{filename}2.jpg')

    # Get matches
    kp1, desc1 = get_AKAZE(img1)
    kp2, desc2 = get_AKAZE(img2)
    matches = find_matches(desc1, desc2, match_ratio=match_ratio)
    print('num_matches:', len(matches))
    match_points = get_match_points(kp1, kp2, matches)

    # Draw matches
    np.random.seed(0) # random color sampling
    matched = draw_matches(img1, img2, match_points)
    save_img(matched, f'q2_result/{filename}_match.jpg')

    # Warp images
    np.random.seed(0) # random sampling for RANSAC
    img1_warped, img2_warped = warp_images(img1, img2, match_points,
                                           ransac_size, ransac_threshold)
    save_img(img1_warped, f'q2_result/{filename}_warped1.jpg')
    save_img(img2_warped, f'q2_result/{filename}_warped2.jpg')

    # Merging 1: Ignore img2 if overlapped
    overlapped1 = overlap_images(img1_warped, img2_warped)
    save_img(overlapped1, f'q2_result/{filename}_overlapped1.jpg')

    # Merging 2: Ignore img1 if overlapped
    overlapped2 = overlap_images(img2_warped, img1_warped)
    save_img(overlapped2, f'q2_result/{filename}_overlapped2.jpg')

    # Merging 3: average if overlapped
    merged = merge_images(img1_warped, img2_warped)
    save_img(merged, f'q2_result/{filename}_merged.jpg')

    # Blending: your custom blending, e.g., alpha, 2-banded, etc.
    # If you want to use more information than the warped images,
    # feel free to input them in kwargs
    ### YOUR CODE HERE  (do not modify this line)
    kwargs = {'alpha': 0.25}
    ### END YOUR CODE   (do not modify this line)

    # Compare merging 3 and your custom blending with a toy example
    toy1_warped = np.zeros([600, 600, 3], dtype=np.uint8)
    toy1_warped[200:500, 300:500] = np.array([255, 127, 127], dtype=np.uint8)
    toy2_warped = np.zeros([600, 600, 3], dtype=np.uint8)
    toy2_warped[100:300, 100:400] = np.array([127, 127, 255], dtype=np.uint8)
    merged = merge_images(toy1_warped, toy2_warped)
    save_img(merged, f'q2_result/{filename}_toy_merged.jpg')
    blended = blend_images(toy1_warped, toy2_warped, **kwargs)
    save_img(blended, f'q2_result/{filename}_toy_blended.jpg')

    # Apply your custom blending to warped images
    blended = blend_images(img1_warped, img2_warped, **kwargs)
    save_img(blended, f'q2_result/{filename}_blended.jpg')

