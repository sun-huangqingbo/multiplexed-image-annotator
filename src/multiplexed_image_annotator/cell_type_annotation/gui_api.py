# uncomment the following lines to run the real code
# from model import Annotator
# import torch
# from utils import gui_run
#
import json
import os
from .model import Annotator
import pandas as pd
import numpy as np


def gui_run(marker_list_path, image_path, mask_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence, n_jobs=0):

    # write image and mask paths to a csv file
    temp = [[image_path, mask_path]]
    pd.DataFrame(temp).to_csv(os.path.join(main_dir, "images.csv"), index=False, header=["image_path", "mask_path"])
    
    path_ = os.path.join(main_dir, "images.csv")
    annotator = Annotator(marker_list_path, path_, device, main_dir, batch_id, strict, infer, min_cells, normalize, blur, amax, confidence, cell_size, cell_type_confidence, n_jobs=n_jobs)
    if not annotator.channel_parser.immune_base and not annotator.channel_parser.immune_extended and not annotator.channel_parser.immune_full and not annotator.channel_parser.struct and not annotator.channel_parser.nerve:
        raise ValueError("No panels are applied. Please check the marker list.")
    annotator.preprocess()
    annotator.predict(bs)
    annotator.generate_heatmap(integrate=True)
    if n_regions > 0:
        annotator.tissue_region_analysis(n_regions)
    annotator.neighborhood_analysis(integrate=True, normalize=True)
    annotator.export_annotations()
    annotator.colorize()
    annotator.cell_type_composition()
    annotator.clear_tmp()

    # delete csv file
    os.remove(path_)

    intensity_dict = {}
    for i in range(len(annotator.preprocessor.intensity_full[0])):
        intensity_dict[i + 1] = annotator.preprocessor.intensity_full[0][i]
    intensity_dict[0] = np.zeros_like(annotator.preprocessor.intensity_full[0][0])
    names = annotator.get_cell_type_names()

    return intensity_dict, names
    

def gui_batch_run(marker_list_path, image_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence, n_jobs=0):
    annotator = Annotator(marker_list_path, image_path, device, main_dir, batch_id, strict, infer, min_cells, normalize, blur, amax, confidence, cell_size, cell_type_confidence, n_jobs=n_jobs)
    if not annotator.channel_parser.immune_base and not annotator.channel_parser.immune_extended and not annotator.channel_parser.immune_full and not annotator.channel_parser.struct and not annotator.channel_parser.nerve:
        raise ValueError("No panels are applied. Please check the marker list.")
    annotator.preprocess()
    annotator.predict(bs)
    annotator.generate_heatmap(integrate=True)
    if n_regions > 0:
        annotator.tissue_region_analysis(n_regions)
    annotator.neighborhood_analysis(integrate=True, normalize=True)
    annotator.export_annotations()
    annotator.colorize()
    annotator.cell_type_composition()
    annotator.clear_tmp()
    

def gui_api(working_addr):
    # read in params from json in the folder ./working_dir_temp/

    with open(f"{working_addr}/hyperparams.json") as f:
        hyperparameters = json.load(f)

    marker_list_path = hyperparameters.get('marker_file')
    image_path = hyperparameters.get('image_file')
    mask_path = hyperparameters.get('mask_file')
    device = hyperparameters.get('device')
    main_dir = hyperparameters.get('main_dir')
    batch_id = "single_run"
    strict = hyperparameters.get('strict')
    infer = hyperparameters.get('infer')
    min_cells = hyperparameters.get('min_cells')
    n_regions = hyperparameters.get('n_regions')
    normalize = hyperparameters.get('normalize')
    blur = hyperparameters.get('blur')
    amax = hyperparameters.get('upper_limit')
    confidence = hyperparameters.get('confidence')
    cell_type_confidence = hyperparameters.get('cell_type_confidence')
    bs = hyperparameters.get('batch_size')
    cell_size = hyperparameters.get('cell_size')

    img, cell_type_names = gui_run(marker_list_path, image_path, mask_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence)

    return img, cell_type_names

def batch_process(working_dir):
    with open(f"{working_dir}/hyperparams_batch.json") as f:
        hyperparameters = json.load(f)

    marker_list_path = hyperparameters.get('marker_file')
    image_path = hyperparameters.get('csv_file')
    device = hyperparameters.get('device')
    main_dir = hyperparameters.get('main_dir')
    batch_id = hyperparameters.get('batch_id')
    strict = hyperparameters.get('strict')
    infer = hyperparameters.get('infer')
    min_cells = hyperparameters.get('min_cells')
    n_regions = hyperparameters.get('n_regions')
    normalize = hyperparameters.get('normalize')
    blur = hyperparameters.get('blur')
    amax = hyperparameters.get('upper_limit')
    confidence = hyperparameters.get('confidence')
    cell_type_confidence = hyperparameters.get('cell_type_confidence')
    bs = hyperparameters.get('batch_size')
    cell_size = hyperparameters.get('cell_size')

    gui_batch_run(marker_list_path, image_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence)
    f = f"{working_dir}/output.txt"
    with open(f, "w") as file:
        file.write("Batch process completed")
