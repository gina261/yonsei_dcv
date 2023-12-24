import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


parser = argparse.ArgumentParser(description='run mystery visualization')
parser.add_argument('ind', help='The index of data to load in [1,4]')


def reconstructImage(IQ, palette):
    """
    Given a quantized image IQ and their value, return a floating point image
    """
    # opencv is full of these assertions.
    # If you're looking here you may have an error.
    # Check to see what the assertion says
    assert(np.issubdtype(IQ.dtype, np.integer))
    return palette[IQ]


def colormapArray(data, colors, ind=None):
    """
    Visualize the data in your own creative way using the given color map

    Inputs:
        data: an HxWxD matrix, where H and W are spatial sizes
        colors: an Nx3 color map of colors in [0,1] [R,G,B]
        ind: index of data in char; you may want to process each differently
    Outputs:
        image: an HxW float64 image using the given colormap
    """
    H, W, D = data.shape
    N, C = colors.shape
    image = np.zeros([H, W, C], dtype=np.float64)
    ### YOUR CODE HERE
    
    v_min_nan = np.min(data[~np.isnan(data)])
    # print(v_min_nan)
    
    # If we data for the NAN data
    # To handle it substitute with v_min values for the NAN values
    data[np.isnan(data)] = v_min_nan
    sorted_val = np.sort(data.reshape(-1))
    
    # Draw histogram to find percentile of the data distribution
    result = plt.hist(sorted_val, bins=500)
    
    # Get the total counted values by the histogram
    tot_val = result[0]
    tot_bin = result[1]
    
    # Get cumulative values for the total distribution
    cumulative_val = np.cumsum(tot_val)

    # Percentile of the total data distribution , parameter to choose proper values
    percentile = 0.995
    threshold = 512 * 512 * 9 * percentile
    # print(threshold)
    outlier = -1
    for idx in range(500):
        if cumulative_val[idx] > threshold:
            outlier = int(tot_bin[idx])
            break
    print(f'Outlier for this data {outlier}')    


    # Step 2 : Handle Outlier values due to disturb data distribution
    data[data>outlier] = outlier
    
    std_data = data.copy().astype('float64')
    epsilon = 1e6
    for ch in range(9):
        cur_channel_mean = np.mean(data[:,:,ch])
        cur_channel_std = np.std(data[:,:,ch])
        # print(data[:,:,ch].dtype)
        std_data[:,:,ch] = ((data[:,:,ch].reshape(-1) - cur_channel_mean) / (cur_channel_std + epsilon)).reshape(512,512)
    if ind == 1:
        std_data[:,:,[0,1,2,3,4,5,6,7]]
    elif ind == 2:
        std_data[:,:,[0,1,2,3,4,5,6,7]]
    elif ind == 3:
        std_data[:,:,[0,1,2,3,4,5,6,7]]
    elif ind == 4:
        std_data[:,:,[0,1,2,3,4,5,6,7]]
    
    
    # Gather Total Channel values
    ch = std_data.copy()
    # ch = np.stack([data[:,:,0],data[:,:,1],data[:,:,4],data[:,:,5], data[:,:,6], data[:,:,7]], axis = 2)
    ch = np.mean(ch, axis=2)
    # data[data> 1000] = 0
    # ch = np.mean(data, axis=2)

    new_pic_idx = np.zeros((512,512), dtype = 'float64')
    new_pic_idx_int = np.zeros((512,512), dtype = 'int64')
    ch_max = ch.max()
    ch_min = ch.min()
    new_pic_idx = (ch - ch_min) / (ch_max - ch_min)
    new_pic_idx = (1023) * new_pic_idx
    new_pic_idx_int =  new_pic_idx.copy().astype('int64')
    image[:,:,:] = colors[new_pic_idx_int, :]
    ### END YOUR CODE
    return image


if __name__ == "__main__":
    # data index
    args = parser.parse_args()
    ind = int(args.ind)

    # load data
    data = np.load(f'q3_data/mysterydata{ind}.npy')

    # load colors
    colors = np.load('q3_data/colors.npy')

    # save your visualizations in q3_results/
    os.makedirs('q3_results', exist_ok=True)

    # play with data before commenting out this.
    #pdb.set_trace()

    # visualize data
    image = colormapArray(data, colors, ind)
    plt.imsave(f'q3_results/mysterydata{ind}.png', image)

