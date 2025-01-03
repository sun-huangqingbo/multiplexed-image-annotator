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
    colors = []
    for i in range(n-1):
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        colors.append([r, g, b])
    colors.append([192, 192, 192])
    return colors

def color_legend(main_dir, colors):
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
        if i == 3 or i == 11:
            text_color ='black'
        else:
            text_color ='white'
        ax.text(col + 0.5, num_rows - row - 1 + 0.5, name, va='center', ha='center', fontsize=10, color=text_color)

    # Remove the axes
    ax.set_xlim(0, num_cols)
    ax.set_ylim(0, num_rows)
    ax.axis('off')

    plt.savefig(os.path.join(main_dir, 'color_legend.png'), bbox_inches='tight')
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


def smooth(mask,c):
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


