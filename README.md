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
- pyqt5
- scipy
- seaborn
- scikit-learn
- skimage
- tifffile
- timm
- torch
- umap-learn
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

For a standard OME-TIFF file, the image channel names (marker names) can be automatically extracted and parsed from its metadata, so no text file is needed. A QPTIFF file is also supported for automated metadata extraction; however, the computer needs to have JAVA installed.

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
- **Image Path**: The path to the TIFF image.
- **Marker File Path**: The path to the text file containing marker names.
- **Mask Path**: The path to the cell segmentation image.
- **Device**: Specify whether to use CPU or GPU for computation.
- **Batch Size**: The batch size used for marker imputation and cell type prediction.
- **Main Directory**: The directory where all output files will be saved.
- **Strict**: Determines whether models with missing markers can be applied. When set to `False`, the model will impute or use blank channels for missing markers. When set to `True`, models with any missing markers cannot be applied. Default: `False`.
- **Infer**: Enables the marker imputation feature. When set to `True`, the model will impute images for missing markers. Default: `True`.
- **Normalize**: Controls whether the multiplexed image should be normalized. Default: `True`. We recommend enabling this to ensure proper normalization for accurate cell type prediction.
- **Blur**: Specifies whether to apply Gaussian blurring during preprocessing. The default value is `0.5`, with an expected range of `0` to `1`.
- **Upper Limit**: Defines the percentile value used as the upper threshold to clip each image channel based on its intensity values.
- **Confidence**: The threshold for determining the validity of a cell type prediction. If the confidence is lower than this value, the model will classify the cell as "Others." The default value is `0.3`. Higher values will result in more "Others" annotations rather than valid cell types.
- **Cell Size**: The estimated cell size in pixels for the query image.
- **Cell Type-Specific Confidence**: Allows for setting confidence thresholds for individual cell types rather than using a unified value. This parameter should be edited directly in the `hyperparameter.json` file and selected in the user interface.
- **min cells** Minimal number of cells in a group for that group to be considered a new cell type. Set it to `-1` to disable the new cell type clustering.
- **n regions** Number of tissue compartment regions to be identified.
- **n_jobs** Number of processes to use for image preprocessing.



All hyperparameters can be pre-determined and saved in the `hyperparameter.json` file. Select this file when using the plugin without re-entering it.


### User interface
Our user interface is realized as a Napari Plugin. It has two modules: 1. Single Image Annotator; 2. Batch Processing. 

Single Image Annotator is designed to tune the hyparameters, annotate one image at a time, and visualize the annotation results. After selecting Single Image Annotator, the napari window will look like follows, by clicking four buttons (in red box) to launch the annotator
![image](https://github.com/user-attachments/assets/0b673af5-45b5-456c-830a-366a73438a8b)



https://github.com/user-attachments/assets/035a587b-a6b1-4364-a0e9-45ec9e93ef93


![napari_plugin](https://github.com/user-attachments/assets/858c9845-3c00-4e18-bc15-88e482be5b59)

The main Napari build-in viewer (yellow box) shows the original tissue image, cell segmentation, and our annotation results in three layers; the panel in red box is for user to required file paths and hyperparameters; below, the panel in cyan lists marker and cell-type names; when the user clicks any cells on the cell segmentation layer in the main viewer, the green panel will show the cell-level marker expression intensity of that cell.

When running the Batch Processing, RIBCA will be running in the backend and do not involve interactive and visualization features. Results will be saved to the assigned folder.

### Contact
sunh at stanford dot edu
