# uncomment the following lines to run the real code
# from model import Annotator
# import torch
# from utils import gui_run
#
import json
import os
from .utils import gui_run, gui_batch_run

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
    normalization = hyperparameters.get('normalize')
    blur = hyperparameters.get('blur')
    confidence = hyperparameters.get('confidence')
    cell_type_confidence = hyperparameters.get('cell_type_confidence')
    bs = hyperparameters.get('batch_size')

    img = gui_run(marker_list_path, image_path, mask_path, device, main_dir, batch_id, bs, strict, normalization, blur, confidence, cell_type_confidence)

    return img

def batch_process(working_dir):
    with open(f"{working_dir}/hyperparams_batch.json") as f:
        hyperparameters = json.load(f)

    marker_list_path = hyperparameters.get('marker_file')
    image_path = hyperparameters.get('csv_file')
    device = hyperparameters.get('device')
    main_dir = hyperparameters.get('main_dir')
    batch_id = hyperparameters.get('batch_id')
    strict = hyperparameters.get('strict')
    normalization = hyperparameters.get('normalize')
    blur = hyperparameters.get('blur')
    confidence = hyperparameters.get('confidence')
    cell_type_confidence = hyperparameters.get('cell_type_confidence')
    bs = hyperparameters.get('batch_size')

    gui_batch_run(marker_list_path, image_path, device, main_dir, batch_id, bs, strict, normalization, blur, confidence, cell_type_confidence)
    f = f"{working_dir}/output.txt"
    with open(f, "w") as file:
        file.write("Batch process completed")
