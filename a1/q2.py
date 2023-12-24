import argparse
import os
import sys
import cv2
import numpy as np
import pdb


def coverPalette(N):
    # Return the palette we're using
    return np.linspace(1, 0, 2 ** N)


def reconstructImage(IQ, palette):
    """
    Given a quantized image IQ and their value, return a floating point image
    """
    # opencv is full of these assertions.
    # If you're looking here you may have an error.
    # Check to see what the assertion says
    assert(np.issubdtype(IQ.dtype, np.integer))
    return palette[IQ]


def upscaleNN(I, target_size):
    """
    NN upsample I until it hits a target size but without going over 4096
    """
    h, w = I.shape[:2]
    scale = 1
    while True:
        if min(h * scale, w * scale) >= target_size:
            break
        if max(h * (scale + 1), w * (scale + 1)) > 4096:
            break
        scale += 1
    # usually you should call a library but providing the call here defeats
    # the purpose :)
    shape = (scale, scale) if I.ndim == 2 else (scale, scale, 1)
    return np.kron(I, np.ones(shape))


def quantize(v, palette):
    """
    Given a scalar v and array of values palette,
    return the index of the closest value
    """
    ind = 0
    ### YOUR CODE HERE
    """
    Check all of the L1 Norm abs function then, check argmin values for v
    Time consuming but could find solution
    """
    
    # For scalar values v
    if v.ndim == 0:
        l1_norm = np.abs(v-palette)
        ind = int(np.argmin(l1_norm))
        
    # For vector values v
    else: 
        new_palatte =  np.vstack([palette for i in range(v.shape[0])])
        v_res = v.reshape((v.shape[0], 1))
        l1_norm = np.abs(v_res - new_palatte)
        ind = np.argmin(l1_norm, axis = 1).astype(int)
    ### END YOUR CODE
    return ind


def quantizeNaive(IF, palette):
    """
    Given a floating-point image return quantized version (Naive)
    """
    IQ = np.zeros_like(IF, dtype=np.uint8)
    ### YOUR CODE HERE
    """
    IQ : Image Quantized
    IF : Image Floating-point 32
    """
    if IF.ndim == 2:
        for y in range(IQ.shape[0]):
            for x in range(IQ.shape[1]):
                IQ[y][x] = quantize(IF[y][x], palette)
                
    # Code for RGB scales
    # Check ndim with 3 channels
    elif IF.ndim == 3:
        for y in range(IQ.shape[0]):
            for x in range(IQ.shape[1]):
                IQ[y,x,:] = quantize(IF[y,x,:], palette)
        
    ### END YOUR CODE
    return IQ


def quantizeFloyd(IF, palette):
    """
    Given a floating-point image return quantized version (Floyd-Steinberg)
    """
    IQ = np.zeros_like(IF, dtype=np.uint8)
    ### YOUR CODE HERE
    
    # Code for gray scales
    # Check ndim with 3 channels
    if IF.ndim == 2:
        ref_IF = IF.copy()
        ref_W = IF.shape[1]
        ref_H = IF.shape[0]
        for y in range(IF.shape[0]):
            for x in range(IF.shape[1]):
                ref_val = ref_IF[y][x].copy()
                q_color_idx = quantize(ref_val, palette)
                IQ[y][x] = q_color_idx 
                q_color = palette[q_color_idx]
                err = ref_val - q_color
                """
                x and y should be changed with pesudo code
                Edge case will be occured with 0 and max values for
                input values IF.shape[0] and IF.shape[1]
                x -> 0 ~ ref_W -1
                y -> 0 ~ ref_H -1
                """
                if x < ref_W -1: 
                    ref_IF[y][x+1] += err * (7/16)
                if 0 < x and y < ref_H - 1:
                    ref_IF[y+1][x-1] += err * (3/16)
                if y < ref_H -1:
                    ref_IF[y+1][x] += err * (5/16)
                if x < ref_W -1 and y < ref_H -1:
                    ref_IF[y+1][x+1] += err * (1/16)
    
    # Code for RGB scales
    # Check ndim with 3 channels
    elif IF.ndim == 3:
        ref_IF = IF.copy()
        ref_W = IF.shape[1]
        ref_H = IF.shape[0]
        for y in range(IF.shape[0]):
            for x in range(IF.shape[1]):
                ref_val = ref_IF[y,x,:].copy()
                q_color_idx = quantize(ref_val, palette)
                IQ[y,x,:] = q_color_idx 
                q_color = palette[q_color_idx]
                err = ref_val - q_color
                """
                x and y should be changed with pesudo code
                Edge case will be occured with 0 and max values for
                input values IF.shape[0] and IF.shape[1]
                x -> 0 ~ ref_W -1
                y -> 0 ~ ref_H -1
                """
                if x < ref_W -1: 
                    ref_IF[y,x+1,:] += err * (7/16)
                if 0 < x and y < ref_H - 1:
                    ref_IF[y+1,x-1,:] += err * (3/16)
                if y < ref_H -1:
                    ref_IF[y+1,x,:] += err * (5/16)
                if x < ref_W -1 and y < ref_H -1:
                    ref_IF[y+1,x+1,:] += err * (1/16)
    
    ### END YOUR CODE
    return IQ


def parse():
    parser = argparse.ArgumentParser(description='run dither')
    parser.add_argument('source')
    parser.add_argument('target')
    parser.add_argument('algorithm', help='What function to call')
    parser.add_argument('--numbits', default=1, type=int,
                        help="""Number of bits to use;
                                you can play with this,
                                but set 1 for the submission.""")
    parser.add_argument('--rgb', action='store_true',
                        help='Whether to keep rgb or grayscale the image')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    if args.algorithm not in globals():
        print('The algorithm is not recognizable')
        sys.exit(1)

    if not os.path.exists(args.target):
        os.mkdir(args.target)

    color_str = 'rgb' if args.rgb else 'grayscale'

    # Get all the images in the directory ending with .jpg
    images = [fn for fn in os.listdir(args.source) if fn.endswith('.jpg')]
    # os.listdir is NOT guaranteed to read things in consistent order!
    images.sort()

    # Get algorithm and the palette
    # This returns the function with the given argument; this is generally
    # Horrible horrible security, but makes it convenient for the assignment
    # and for you to debug
    algo_fn = globals()[args.algorithm]
    # This is just an array of values
    palette = coverPalette(args.numbits)

    # Run algorithm over images
    for ind, image in enumerate(images):
        print('%d/%d' % (ind, len(images)))

        # Load the image; using opencv we'll usually have to provide uint8
        I = cv2.imread(os.path.join(args.source, image))

        # Convert to [0, 1]
        I = I.astype(np.float64) / 255.
        if not args.rgb:
            I = np.mean(I, axis=2)

        # Call the algorithm and reconstruct the image using the palette
        IQ = algo_fn(I, palette)
        R = reconstructImage(IQ, palette)

        # As a sanity check, we'll write back whatever image we get (including
        # whether we grayscaled it)
        # Beware that you shouldn't tamper with images that you get passed!
        # Note the *255!
        I_path = os.path.join(args.target,
            f'{os.path.splitext(image)[0]}_{color_str}_orig.png')
        cv2.imwrite(I_path, (I * 255.).astype(np.uint8))
        R_path = \
            f'{os.path.splitext(image)[0]}_{color_str}_{args.algorithm}.png'
        R_path = os.path.join(args.target, R_path)
        cv2.imwrite(R_path, (R * 255.).astype(np.uint8))

