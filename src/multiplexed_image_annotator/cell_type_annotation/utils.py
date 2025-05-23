# -*- coding: utf-8 -*-
from tqdm import tqdm
from tifffile import imwrite
import numpy as np
import os
from skimage import filters
from skimage.morphology import dilation, disk
from skimage.io import imread
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import random
from skimage.transform import resize

def number_to_rgb(value, cmap_name='viridis'):
    if value < 0 or value > 1:
        raise ValueError("Value must be between 0 and 1")

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Get the RGB color code
    rgb = cmap(norm(value))[:3]  # Ignore the alpha value

    rgb_255 = list(int(x * 255) for x in rgb)

    return rgb_255

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def get_colors(n):
    """
    Generate n visually distinct colors similar to the standard palette.
    
    Args:
        n: Number of colors to generate
        
    Returns:
        List of (r,g,b) tuples with values in range 0-255
    """
    n = n - 1  # Adjust for the last color which is always gray
    colors = []
    
    # Start with the standard palette for smaller n
    standard_colors = [
        (255, 0, 0),      # Red
        (0, 0, 255),      # Blue
        (0, 128, 0),      # Green
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 165, 0),    # Orange 
        (128, 0, 128),    # Purple
        (0, 128, 128),    # Teal
        (128, 0, 0),      # Maroon
        (0, 0, 128),      # Navy
        (128, 128, 0),    # Olive
        (255, 192, 203),  # Pink
        (165, 42, 42),    # Brown
        (0, 255, 0),      # Lime
        (135, 206, 235),  # Sky Blue
        (75, 0, 130),     # Indigo
        (255, 215, 0),    # Gold
        (192, 192, 192)   # Silver
    ]
    
    # If n is less than or equal to the standard palette size, just return the first n colors
    if n <= len(standard_colors):
        colors = standard_colors[:n]
        # Add gray color at the end
        colors.append((192, 192, 192))
        return colors
    
    # If we need more colors, use HSV color space to generate them evenly around the color wheel
    
    # First, add all standard colors
    colors = standard_colors.copy()
    
    # Then generate additional colors
    remaining = n - len(colors)
    
    # Use golden ratio to get well-distributed colors
    golden_ratio_conjugate = 0.618033988749895
    h = 0.1  # Starting hue
    
    # Generate saturation and value ranges similar to the existing palette
    saturations = [0.7, 0.8, 0.9, 1.0]
    values = [0.7, 0.8, 0.9, 1.0]
    
    while len(colors) < n:
        # Use golden ratio to get next hue - this creates visually pleasing distribution
        h = (h + golden_ratio_conjugate) % 1.0
        
        # Cycle through different saturation and value combinations for variety
        s = saturations[len(colors) % len(saturations)]
        v = values[len(colors) % len(values)]
        
        # Convert HSV to RGB (0-1 range)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to 0-255 range and append to colors list
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    colors.append((192, 192, 192))
    
    return colors


def color_legend(main_dir, colors, cell=True):
      # Calculate the number of rows and columns
    num_colors = len(colors)
    num_cols = 6  # Set the number of columns
    num_rows = (num_colors + num_cols - 1) // num_cols  # Calculate the number of rows needed
    
    # Create a plot for the color legend in multiple rows
    fig, ax = plt.subplots(figsize=(2.5 * num_cols, 0.4 * num_rows))
    # Create the color legend in a grid layout
    for i, name in enumerate(colors):
        row = i // num_cols
        col = i % num_cols
        ax.add_patch(plt.Rectangle((col, num_rows - row - 1), 1, 1, color=colors[name]))
        hex = colors[name]

        rgb = [int(hex[i:i + 2], 16) for i in (1, 3, 5)]
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        text_color = 'black' if luminance > 0.5 else 'white'
        ax.text(col + 0.5, num_rows - row - 1 + 0.5, name, va='center', ha='center', fontsize=10, color=text_color)

    # Remove the axes
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.axis('off')

    if cell:
        plt.savefig(os.path.join(main_dir, 'cell_color_legend.png'), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(main_dir, 'tissue_region_color_legend.png'), bbox_inches='tight')
    plt.close()


def get_void_vote():
    return {"CD4 T cell": 0, "CD8 T cell": 0, "Dendritic cell": 0, "B cell": 0, "M1 macrophage cell": 0, 
                "M2 macrophage cell": 0, "Regulatory T cell": 0, "Granulocyte cell": 0, "Plasma cell": 0, "Natural killer cell": 0, "Mast cell": 0,
                "Stroma cell": 0 , "Smooth muscle": 0, "Endothelial cell": 0, "Epithelial cell": 0, "Proliferating/tumor cell": 0, "Nerve cell": 0}

# This function crops cell with cell_id:position_index dictionary
def crop_cell(dict,img,mask,save_path,file_name,cell_index=None, channel_index=None,margin=12,patch_size=40,save_tensor=False,csv_save_path=None):
    if csv_save_path:
        csv_path = csv_save_path
    else:
        csv_path = save_path+'csv_file/'
    csv_file = file_name.replace("tiff","csv")
    csv_file_path = os.path.join(csv_path, csv_file)
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    min_val = np.min(img,axis=(1,2),keepdims=True)
    img_zero = img-min_val
    
    with open(csv_file_path, 'w', encoding='UTF8') as csvfile:
      csvfile.write("marker_path")
      csvfile.write(",")

      csvfile.write("channel_index")
      csvfile.write("\n")
      # count = 0
      if cell_index is None:
          cell_index = dict.keys()

      for c in tqdm(cell_index):
        
        x_mean = (min(dict[c][0])+max(dict[c][0]))//2
        xmin = x_mean-patch_size/2
        xmin = int(max(xmin,0))
        xmax = int(min(x_mean+patch_size/2,img.shape[1]))
        if xmax-xmin < patch_size:
            xmax = xmin+patch_size

        y_mean = (min(dict[c][1])+max(dict[c][1]))//2
        ymin = y_mean-patch_size/2
        ymin = int(max(ymin,0))
        ymax = int(min(y_mean+patch_size/2,img.shape[2]))
        if ymax-ymin < patch_size:
            ymax = ymin+patch_size

        img_patch = np.zeros((img.shape[0],patch_size,patch_size))
        mask_patch = np.zeros((mask.shape[0],patch_size,patch_size))
        img_zero_patch = np.zeros((img.shape[0],patch_size,patch_size))
        img_patch[:,:(xmax-xmin),:(ymax-ymin)] = img[:,xmin:xmax,ymin:ymax]
        img_zero_patch[:,:(xmax-xmin),:(ymax-ymin)] = img_zero[:,xmin:xmax,ymin:ymax]
        mask_patch[:,:(xmax-xmin),:(ymax-ymin)] = mask[:,xmin:xmax,ymin:ymax]

        mask_smooth = smooth(mask_patch[0],c)
        
        # create a csv file listing all the cell_mask index

        marker_a = img_zero_patch*mask_smooth
        marker_a = marker_a+min_val

        #mask_smooth = np.vstack((mask_smooth,mask_patch[0]))
        marker_path = save_path+'marker_patch/'+file_name
        if not os.path.exists(marker_path):
          os.makedirs(marker_path)


        if save_tensor:
            f = os.path.join(marker_path, r"{}.pt".format(c))
            tensor_image = torch.tensor(marker_a, dtype=torch.float32)
            torch.save(tensor_image, f)
            a = tensor_image.numpy()
            diff = np.sum(a - marker_a)

            assert diff < 1e-4
            
        else:
            f = os.path.join(marker_path, r"{}.tiff".format(c))
            imwrite(f, marker_a)


        csvfile.write(f)
        csvfile.write("\n")


def crop_cell(image, mask, min_val, c_id, pos_dict, patch_size):
    x_mean = (min(pos_dict[c_id][0]) + max(pos_dict[c_id][0])) // 2
    xmin = x_mean - patch_size / 2
    xmin = int(max(xmin, 0))
    xmax = int(min(xmin + patch_size, image.shape[1]))

    y_mean = (min(pos_dict[c_id][1]) + max(pos_dict[c_id][1])) // 2
    ymin = y_mean - patch_size / 2
    ymin = int(max(ymin, 0))
    ymax = int(min(ymin + patch_size, image.shape[2]))

    img_patch = np.zeros((image.shape[0], patch_size, patch_size))
    mask_patch = np.zeros((patch_size, patch_size))
    img_zero_patch = np.zeros((image.shape[0], patch_size, patch_size))
    img_patch[:, :(xmax-xmin), :(ymax-ymin)] = image[:, xmin:xmax, ymin:ymax]
    img_zero_patch[:, :(xmax-xmin), :(ymax-ymin)] = image[:, xmin:xmax, ymin:ymax]
    mask_patch[:(xmax-xmin), :(ymax-ymin)] = mask[xmin:xmax, ymin:ymax]

    mask_smooth = smooth(mask_patch, c_id)
    
    marker_a = img_zero_patch * mask_smooth
    marker_a = marker_a + min_val

    avg_int = np.zeros(image.shape[0])
    # get avg intensity of the cell for each channel
    for i in range(image.shape[0]):
        avg_int[i] = np.mean(marker_a[i, :, :][mask_patch > 0])
    return marker_a, avg_int

def smooth(mask, c):
    mask = mask == c
    smooth = mask.astype("f")
    count = 1
    for j in range (1, 5, 1):
        mask_dilated = dilation(mask, disk(j))

        smooth += mask_dilated.astype("f")
        count += 1
        for i in np.arange(0, j-1, 1):
            smooth += filters.gaussian(mask_dilated, sigma=1+i)
            count += 1
    smooth /=  count
    smooth /= np.max(smooth+1e-6)

    return smooth

def process_chunk(chunk_data):
    # Process a chunk of the mask
    start_row, end_row, mask_chunk = chunk_data
    local_dict = {}
    
    for i in range(mask_chunk.shape[0]):
        for j in range(mask_chunk.shape[1]):
            c = mask_chunk[i, j]
            if c == 0:  # 0 is background
                continue
                
            if c not in local_dict:
                local_dict[c] = ([], [])
            
            # Adjust i to global coordinates
            local_dict[c][0].append(i + start_row)
            local_dict[c][1].append(j)
    
    return local_dict


# Create a local version of the worker function that processes batches
def process_cell_batch(cell_batch, img_zero_local, mask_local, min_val_local, 
                    shared_cell_pos_dict, patch_size_local, channel_index_local):
    results = []
    for cell_idx, cell_id in cell_batch:
        patch, avg_int = crop_cell(img_zero_local, mask_local, min_val_local, 
                                cell_id, shared_cell_pos_dict, patch_size_local)
        
        avg_int_copy = avg_int.copy()
        # rescale
        patch = resize(patch, (patch.shape[0], 40, 40), anti_aliasing=True, order=0, preserve_range=True)
        
        if -1 in channel_index_local:
            # get index
            index = list(channel_index_local).index(-1)
            # temporary remove -1 from channel_index
            channel_index_ = np.delete(channel_index_local, index)
            patch = patch[channel_index_, :, :]
            # concat
            blank_patch = -np.ones_like(patch[0:1])
            patch = np.concatenate((patch[:index], blank_patch, patch[index:]), axis=0)
        else:
            patch = patch[channel_index_local, :, :]

            
        results.append((cell_idx, patch, avg_int))
    return results
