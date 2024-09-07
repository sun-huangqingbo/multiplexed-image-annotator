# A generic cell type annotator for highly multiplexed tissue images

## Overview
This repository contains a flexible and robust cell-type annotation tool for highly multiplexed tissue images. Our paper can be found [here]. It is generalizable to new image collections without extra fine-tuning. 
Instead of a single model, we constructed an ensemble of image-derived models, which makes it compatible with any common antibody panel by matching their markers with one or multiple base models in this ensemble.
The output of this tool is a cell-type map, its annotation confidence, and spatial statistics of cell-type distribution. Our software has a Napari plugin for interactively validating annotations.

## Citation


## Requirements
- matplotlib
- napari
- numpy
- seaborn
- skimage
- tifffile
- torch
- umap

NVIDIA graphics hardware with CUDA and cuDNN support are recommended. 

## Getting started
### Installation
Clone this repo and install it locally by running the following
```bash
pip install -e .
```

### Required input files and hyper-parameters
Our tool requires a 3D (CHW) multiplexed tissue TIFF image stack, its cell segmentation mask (in 2D) where 0 means background and 1 ~ N means cell regions, and a text file containing antobody panel where each line listed a marker name. Please find exmaples in `example_files` folder.

Note that the antibody/marker names need to EXACTLY match the names provided below in order to let the program to automatically match them with our panels.
MARKER NAMES

For batch processing, it needs an additional csv file with two columns listed the paths of images and their segmentation masks per row. The heads of two columns are: `image_paths` and `mask_paths`. Please find exmaples in `example_files` folder.

Our pipeline requires the following hyper-parameters:
- image path: path of the TIFF image
- marker file path: path of the text file containing marker names
- mask path: path of the cell segmentation image
- device: use cpu or gpu for computing
- batch size: used for marker imputation and cell type prediction
- main directory: the directory that all output files will be saved at
- impute: whether to use the marker imputation feature, default True
- normalize: whether to normalize the multiplexed image, default True. We recommend to use this feature to ensure the image is normalized as expected, which is essential for cell type prediction
- blur: whether to perform Gaussian blurring of the image in the preprocessing step, default value is set to 0.5. The range of this value is expected to be 0 ~ 1.
- confidence: the threshold used to determine if a cell type call is valid; if the prediction confidence is lower than this value, the model will predict that cell image as "Others", default value is set to 0.3. Larger this value, more "Others" type will be annotated rather than valid cell types
- cell type specific confidence: set threshold for each cell type explicitly rather than a unified confidence threshold above. To set this hyper-parameter, please directly edit it in the `hyperparameter.json` and select this file in the user interface.


### User interface
Our user interface is realized as a Napari Plugin.

