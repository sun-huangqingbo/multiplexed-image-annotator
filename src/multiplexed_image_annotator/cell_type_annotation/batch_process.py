import os
import json
import shutil
import time

def batch_process_func(working_dir):
    print(os.getcwd())
    with open(f"{working_dir}/hyperparams_batch.json") as f:
        params = json.load(f)

    # real run
    # mask = gui_run(marker_file, image_file, None, device, main_dir, "test", strict, normalize, blur, confidence, None)

    # mock run (simulation of the real run)
    print("worker received:")
    print(params)
    # copy a image file to the working_dir_temp
    shutil.copyfile("./src/multiplexed_image_annotator/cell_type_annotation/imgs/test_colorized_annotation_1.png", f"{working_dir}/output_img.png")
    time.sleep(10)