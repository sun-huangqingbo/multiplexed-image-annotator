# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
import pickle

from tifffile import imwrite
from skimage.io import imread
from skimage.morphology import dilation, disk
from skimage import filters
from skimage.transform import resize
import torch

from .markerParse import MarkerParser
from .markerImputer import MarkerImputer


class ImageProcessor(object):
    def __init__(self, csv_path, parser, main_path, device, batch_id='', infer=True, normalization=True, blur=0, amax=100, cell_size=30, logger=None) -> None:
        df = pd.read_csv(csv_path)
        self.image_paths = df['image_path']
        self.mask_paths = df['mask_path']
        assert len(self.image_paths) == len(self.mask_paths)

        self.logger = logger

        self._n_images = len(self.image_paths)

        self.logger.log("Number of images: {}.".format(self._n_images))

        self.main_dir = main_path
        self.save_path = os.path.join(self.main_dir, "tmp")

        self.batch_id = batch_id

        self.normalization = normalization
        self.blur = blur
        self.amax = amax

        self.parser = parser

        self.cell_pos_dict = []
        # self.intensity_all = {}
        self.intensity_full = []
        # for marker in self.parser.markers:
        #     self.intensity_all[marker] = []

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.infer = infer
        self.masks = []
        self.device = device
        self.scale = cell_size / 30.0


    def _img2patches(self, image, mask, channel_index, cell_pos_dict, imputer, id, patch_size=40, save_tensor=True, save_path=None, int_full=False):
        min_val, img_zero = self._move_image_range(image)

        patch_size = int(patch_size * self.scale)

        temp = np.zeros((len(cell_pos_dict), len(channel_index), 40, 40))
        cell_index = cell_pos_dict.keys()
        # intensity_all = []
        if int_full:
            intensity_full = []
        for j, c in enumerate(cell_index):
            patch, avg_int = self._crop_cell(img_zero, mask, min_val, c, cell_pos_dict, patch_size)
            # rescale
            patch = resize(patch, (patch.shape[0], 40, 40), anti_aliasing=True, order=0, preserve_range=True)
            if int_full:
                intensity_full.append(avg_int)
            if -1 in channel_index:
                # get index
                index = list(channel_index).index(-1)
                # temporary remove -1 from channel_index
                channel_index_ = np.delete(channel_index, index)
                patch = patch[channel_index_, :, :]
                # concat
                blank_patch = -np.ones_like(patch[0:1])
                patch = np.concatenate((patch[:index], blank_patch, patch[index:]), axis=0)
                avg_int = avg_int[channel_index_]
                avg_int = np.insert(avg_int, index, -1)
            else:
                patch = patch[channel_index, :, :]
                avg_int = avg_int[channel_index]
            temp[j] = patch
            # intensity_all.append(avg_int)

        
        tensor_patch = torch.tensor(temp, dtype=torch.float32)
        del temp
        if imputer is not None:
            tensor_patch = imputer.impute(tensor_patch, 64)
        if save_tensor:
            f = os.path.join(save_path, r"{}.pt".format(id))
            torch.save(tensor_patch, f)
        # intensity_all = np.array(intensity_all).T
        # # normalize to 0-1
        # intensity_all += 1
        # intensity_all /= 2

        if int_full:
            intensity_full = np.array(intensity_full)
            intensity_full += 1
            intensity_full /= 2
            return intensity_full
        return None
    




    def _move_image_range(self,image):
        # move image intensity range to (0,N)
        min_val = np.min(image,axis=(1, 2), keepdims=True)
        img_zero = image - min_val
        return min_val, img_zero
    
    def _crop_cell(self, image, mask, min_val, c_id, pos_dict, patch_size):
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

        mask_smooth = self._smooth(mask_patch, c_id)
        
        marker_a = img_zero_patch * mask_smooth
        marker_a = marker_a + min_val

        avg_int = np.zeros(image.shape[0])
        # get avg intensity of the cell for each channel
        for i in range(image.shape[0]):
            avg_int[i] = np.mean(marker_a[i, :, :][mask_patch > 0])
        return marker_a, avg_int

    def _cell_pos_dict(self, mask):
        # A function to create a dictionary of cell_id:position_index
        # mask is a 2D image, where each cell has a unique id (1,2,....,N)
        # returns a dictionary with cell_id as key and position_index(tuple:(y,x)) as value
        cell_pos_dict = {}
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                c = mask[i,j]
                if c == 0:
                    # 0 is background
                    continue
                if c not in cell_pos_dict:
                    cell_pos_dict[c] = ([], [])

                cell_pos_dict[c][0].append(i)
                cell_pos_dict[c][1].append(j)

        return cell_pos_dict

    def _smooth(self, mask,c):
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
    

    def _normalize(self, img, blur=0, amax=100):
        img = img.astype(np.float32)
        for i in range(img.shape[0]):

            # subtract background
            bg = gaussian_filter(img[i, :, :], sigma=20)
            bg = np.where(bg > 125, 125, bg)
            # bg = restoration.rolling_ball(img[i, :, :], radius=20)
            img[i, :, :] = np.clip(img[i, :, :] - bg, 0, None)


            if blur:
                img[i, :, :] = gaussian_filter(img[i, :, :], sigma=blur)            

            idxx = np.where(img[i, :, :] > 0)
            if len(idxx[0]) == 0:
                img[i, :, :] = -1
                continue
            thresh = np.percentile(img[i, :, :], amax)
            # print(thresh)
            if thresh > 20:
                img[i, :, :] = np.clip(img[i, :, :], 0, thresh)


            img[i, :, :] = 2 * (img[i, :, :] / max(25, np.max(img[i, :, :]))) - 1
        return img

    def transform(self):
        i = 0
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            image = imread(image_path)

            mask = imread(mask_path)
            if len(mask.shape) == 3:
                mask = mask[:, :, 0] # assume the first channel is the mask


            if self.normalization:
                image = self._normalize(image, blur=self.blur, amax=self.amax)
            
            self.masks.append(mask)
            
            cell_pos_dict = self._cell_pos_dict(mask)
            self.cell_pos_dict.append(cell_pos_dict)
            q = 0
            for panel in self.parser.panels:
                if self.parser.indices[panel] is None:
                    continue
                index = self.parser.indices[panel]


                # get index of -1 in index
                idx = [i for i, x in enumerate(index) if x != -1]
                if not self.infer or -1 not in index or panel == "structure" or panel == "nerve":
                    intensity_full = self._img2patches(image, mask, index, cell_pos_dict, None, id=self.batch_id + "_" + str(i) + "_" + panel, 
                                    save_path=self.save_path, save_tensor=True, int_full=q==0)
                else:
                    imputer = MarkerImputer(idx, self.device, panel)
                    print("Imputer for {} is created".format(panel))
                    msg = "Imputer for {} is created. Marker(s) ".format(panel)
                    for ii in range(len(idx)):
                        if index[ii] == -1:
                            msg += "{} ".format(self.parser.panels[panel][ii])
                    msg += "are imputed."
                    self.logger.log(msg)
                    intensity_full = self._img2patches(image, mask, index, cell_pos_dict, imputer, id=self.batch_id + "_" + str(i) + "_" + panel, 
                                    save_path=self.save_path, save_tensor=True, int_full=q==0)


                if q == 0:
                    self.intensity_full.append(intensity_full)
                # for j, m in enumerate(self.parser.panels[panel]):
                #     if len(self.intensity_all[m]) < i + 1:
                #         self.intensity_all[m].append(intensity_all[j])
                q += 1
            i += 1
