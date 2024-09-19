# RIBCA: A generic cell type annotator for highly multiplexed tissue images

## Overview
This repository contains a flexible and robust cell-type annotation tool for highly multiplexed tissue images. Our paper can be found [here](https://www.biorxiv.org/content/10.1101/2024.09.12.612510v1). It is generalizable to new image collections without extra fine-tuning. 
Instead of a single model, we constructed an ensemble of image-derived models, which makes it compatible with any common antibody panel by matching their markers with one or multiple base models in this ensemble.
The output of this tool is a cell-type map, its annotation confidence, and spatial statistics of cell-type distribution. Our software has a Napari plugin for interactively validating annotations.

## Citation
```
@article {Sun2024.09.12.612510,
	author = {Sun, Huangqingbo and Yu, Shiqiu and Casals, Anna Martinez and B{\"a}ckstr{\"o}m, Anna and Lu, Yuxin and Lindskog, Cecilia and Lundberg, Emma and Murphy, Robert F.},
	title = {Flexible and robust cell type annotation for highly multiplexed tissue images},
	elocation-id = {2024.09.12.612510},
	year = {2024},
	doi = {10.1101/2024.09.12.612510},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/09/16/2024.09.12.612510},
	journal = {bioRxiv}
}
```

## Requirements
- magicgui
- matplotlib
- napari
- numpy
- pyqt
- scipy
- seaborn
- skimage
- tifffile
- timm
- torch
- umap
- gdown

NVIDIA graphics hardware with CUDA and cuDNN support is recommended. 

## Getting started
### Installation
Download all models by running
```bash
python download_models.py
```
or directly download them from the URLs in the script and put them in the `src/multiplexed_image_annotator/cell_type_annotation/models` folder.

Clone this repo and install it locally by running the following
```bash
pip install .
```

### Required input files and hyper-parameters
Our tool requires a 3D (CHW) multiplexed tissue TIFF image stack, its cell segmentation mask (in 2D) where 0 means background and 1 ~ N means cell regions, and a text file containing an antibody panel where each line lists a marker name. Please find examples in `example_files` folder.

Note that the antibody/marker names need to EXACTLY match the names provided below in order to let the program to automatically match them with our panels.

- Basic Panel:
CD45, CD20 (or CD79), CD4, CD8, DAPI, CD11c, CD3
 
- Full Panel:
DAPI, CD3, CD4, CD8, CD11c, CD15, CD20 (or CD79), CD45, CD56, CD68, CD138 (or CD38), CD163, FoxP3, Granzyme B, Tryptase
 
- Extended Panel:
DAPI, CD3, CD4, CD8, CD11c, CD20 (or CD79), CD45, CD68, CD163, CD56

- Structure Panel:
DAPI, aSMA, CD31, PanCK, Vimentin, Ki67, CD45

- Nerve Panel:
DAPI, CD45, GFAP (or CHGA)

For batch processing, it needs an additional csv file with two columns listed the paths of images and their segmentation masks per row. The heads of two columns are: `image_paths` and `mask_paths`. Please find exmaples in `example_files` folder.

Our pipeline requires the following hyper-parameters:
- image path: the path of the TIFF image
- marker file path: the path of the text file containing marker names
- mask path: the path of the cell segmentation image
- device: use CPU or GPU for computing
- batch size: used for marker imputation and cell type prediction
- main directory: the directory that all output files will be saved at
- strict: whether to use the marker imputation feature, when it is False, the model will impute images of missing markers. default False.
- normalize: whether to normalize the multiplexed image, default True. We recommend using this feature to ensure the image is normalized as expected, which is essential for cell type prediction
- blur: whether to perform Gaussian blurring of the image in the preprocessing step, the default value is set to 0.5. The range of this value is expected to be 0 ~ 1.
- confidence: the threshold used to determine if a cell type call is valid; if the prediction confidence is lower than this value, the model will predict that the cell image is "Others", the default value is set to 0.3. The larger this value, the more "Others" types will be annotated rather than valid cell types
- cell type specific confidence: set threshold for each cell type explicitly rather than a unified confidence threshold above. To set this hyper-parameter, please directly edit it in the `hyperparameter.json` and select this file in the user interface.

All hyperparameters can be pre-determined and saved in the `hyperparameter.json` file. Select this file when using the plugin without re-entering it.


### User interface
Our user interface is realized as a Napari Plugin. It has two modules: 1. Single Image Annotator; 2. Batch Processing. 

Single Image Annotator is designed to tune the hyparameters, annotate one image at a time, and visualize the annotation results. After selecting Single Image Annotator, the napari window will look like follows, by clicking four buttons (in red box) to launch the annotator
![image](https://github.com/user-attachments/assets/0b673af5-45b5-456c-830a-366a73438a8b)


![napari_plugin](https://github.com/user-attachments/assets/858c9845-3c00-4e18-bc15-88e482be5b59)

The main Napari build-in viewer (yellow box) shows the original tissue image, cell segmentation, and our annotation results in three layers; the panel in red box is for user to required file paths and hyperparameters; below, the panel in cyan lists marker and cell-type names; when the user clicks any cells on the cell segmentation layer in the main viewer, the green panel will show the cell-level marker expression intensity of that cell.

When running the Batch Processing, RIBCA will be running in the backend and do not involve interactive and visualization features. Results will be saved to the assigned folder.

### Contact
sunh at stanford dot edu
