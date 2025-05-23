"""
This module contains four napari widgets declared in
different ways:

- a pure Python function flagged with `autogenerate: true`
    in the plugin manifest. Type annotations are used by
    magicgui to generate widgets for each parameter. Best
    suited for simple processing tasks - usually taking
    in and/or returning a layer.
- a `magic_factory` decorated function. The `magic_factory`
    decorator allows us to customize aspects of the resulting
    GUI, including the widgets associated with each parameter.
    Best used when you have a very simple processing task,
    but want some control over the autogenerated widgets. If you
    find yourself needing to define lots of nested functions to achieve
    your functionality, maybe look at the `Container` widget!
- a `magicgui.widgets.Container` subclass. This provides lots
    of flexibility and customization options while still supporting
    `magicgui` widgets and convenience methods for creating widgets
    from type annotations. If you want to customize your widgets and
    connect callbacks, this is the best widget option for you.
- a `QWidget` subclass. This provides maximal flexibility but requires
    full specification of widget layouts, callbacks, events, etc.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory, magicgui
from magicgui.widgets import CheckBox, Container, create_widget, Label
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLabel, QVBoxLayout
from napari.utils.notifications import show_info
from tifffile import imread
import imageio
import pathlib
import os
import json
import shutil

from .cell_type_annotation import gui_api
from .cell_type_annotation.gui_api import batch_process

import numpy as np

import napari

import xml.etree.ElementTree as ET

import tifffile

import subprocess

import re

import pandas as pd

class BatchProcess(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.params_panel.json_file.changed.connect(self.parse_json)
        self.params_panel.csv_file.changed.connect(self.parse_csv)
        self.params_panel.marker_file.changed.connect(self.add_marker)

        self.params_panel.job_status.bind(self.update_status_txt)
        self.params_panel.call_button.bind(self.update_call_btn)
        self.setLayout(QHBoxLayout())
        # set the title
        self.setWindowTitle("Table for Labels and Corresponding Indices")

        self.label_txt = QLabel("No markers found.")
        self.layout().addWidget(self.label_txt)
        self.viewer.window.add_dock_widget(
            self.params_panel,
            name="Batch Processor",
        )
        self.working_dir_addr = os.path.join(
            os.getcwd(),
            "src/multiplexed_image_annotator/cell_type_annotation/_working_dir_temp/"
        )

        self.hyper_params = dict()


    # after the segmentation algo is done, tell the reader
    def afterwork(self):
        working_dir = pathlib.Path(self.working_dir_addr)
        output_addr = f"{working_dir}/output.txt"

        try: 
            with open(output_addr, 'r') as f:
                lines = f.readlines()
            show_info("Your output_img has been detected!")
        except:
            print("output reading error")
            show_info("Notice! Your output.txt reading error! The process may have problem.")
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        self.update_status_txt("You do not have running jobs.")
        self.update_call_btn(True)
        show_info("Your job is done!")



    # the main function to launch the seg algo
    def seg_algo_caller(self):
        addr_working = pathlib.Path(self.working_dir_addr)
        if os.path.exists(addr_working):
            shutil.rmtree(addr_working)
        print(f"addr_working: {addr_working}")
        # os.mkdir(addr_working)
        os.mkdir(addr_working)
        with open(f"{addr_working}/hyperparams_batch.json", 'w') as f:
            json.dump(self.hyper_params, f)
        self.update_status_txt("Your job is running, please wait for the result.")
        batch_process(addr_working)


    # constructing the UI
    @magicgui(
        call_button="Set Parameters and Run Annotator",
        layout='vertical',
        device={'choices': ['cpu', 'cuda']},
        batch_size={'min': 1, 'max': 10000, 'step': 1},
        blur={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
        upper_limit={'widget_type': 'FloatSlider', 'min': 95, 'max': 100},
        cell_size={'min': 1, 'max': 100},
        min_cells={'min': -1, 'max': 10000},
        n_regions={'min': -1, 'max': 50},
        confidence={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
        main_dir={'widget_type': 'FileEdit', 'mode': 'd'},
        job_status={'widget_type': 'Label', 'value': 'You do not have running jobs.'}
    )
    def params_panel(
        self,
        csv_file=pathlib.Path('PLEASE SELECT YOUR .CSV FILE (REQUIRED)'),
        json_file=pathlib.Path('PLEASE SELECT YOUR PARAMS JSON FILE (OPTIONAL)'),
        marker_file=pathlib.Path('PLEASE SELECT YOUR MARKER FILE (REQUIRED)'),
        batch_id='',
        device='cuda',
        batch_size=128,
        main_dir=pathlib.Path('PLEASE SELECT YOUR MAIN DIR (REQUIRED)'),
        strict=False,
        infer=True,
        min_cells=50,
        n_regions=5,
        normalize=True,
        upper_limit=99.8,
        blur=0.4,
        confidence=0.3,
        cell_size=30,
        job_status='You do not have running jobs.',
        cell_type_confidence=None,
    ):
        new_dict = {
            'json_file': str(json_file),
            'marker_file': str(marker_file),
            'csv_file': str(csv_file),
            'device': device,
            'batch_size': batch_size,
            'main_dir': str(main_dir),
            'strict': strict,
            'infer': infer,
            'min_cells': min_cells,
            'n_regions': n_regions,
            'normalize': normalize,
            'blur': blur,
            'upper_limit': upper_limit,
            'confidence': confidence,
            'batch_id': batch_id,
            'cell_size': cell_size,
            'cell_type_confidence': cell_type_confidence,
            'n_jobs': 0
        }
        self.hyper_params = new_dict

        is_valid = True
        # check whether the image and marker files exist
        if not os.path.exists(marker_file):
            print("marker file does not exist")
            show_info("Notice! Your marker file does not exist!")
            is_valid = False

        # check whether marker file is a txt file
        if not str(marker_file.suffix) == '.txt':
            print("marker file is not a txt file")
            show_info("Notice! Your marker file is not in right format (.txt expected)!")
            is_valid = False

        if not os.path.exists(csv_file):
            print("csv file does not exist")
            show_info("Notice! Your csv file does not exist!")
            is_valid = False

        # check whether input file is a csv file
        if not str(csv_file.suffix) == '.csv':
            print("marker file is not a csv file")
            show_info("Notice! Your csv file is not in right format (.csv expected)!")
            is_valid = False

        # check whether the main_dir exists and is a directory
        if not os.path.exists(main_dir):
            print("main_dir does not exist")
            show_info("Notice! Your main_dir does not exist!")
            is_valid = False
        elif not os.path.isdir(main_dir):
            print("main_dir is not a directory")
            show_info("Notice! Your main_dir is not a directory!")
            is_valid = False

        if is_valid:
            # start the job
            show_info("Your job has been submitted!")
            self.update_status_txt("Your job has been submitted!")
            self.update_call_btn(False)
            worker_seg_algo = napari.qt.threading.create_worker(
                self.seg_algo_caller,
            )

            worker_seg_algo.start()
            
            worker_seg_algo.finished.connect(self.afterwork)

        else:
            self.update_status_txt("Please pay attention to your params!")

    # NOTE: This is the callback function when user input some string
    def user_input_str_callback(self):
        print(self.params_panel.batch_id.value)

    def parse_csv(self):
        try:
            df = pd.read_csv(self.params_panel.csv_file.value)
            first_image_path = df['image_path'].iloc[0]
            try:
                img = imread(first_image_path)
            
            except Exception as e:
                print(f"image reading error as: \n{e}")
                show_info("Notice! Input image from the csv is invalid!")
                return

            if str(first_image_path).endswith('.tiff') or str(first_image_path).endswith('.tif'):
                try:
                    with tifffile.TiffFile(first_image_path) as tif:
                        # print("OME metadata:", tif.ome_metadata)
                        ome_xml_str = tif.ome_metadata
                    # Parse the XML
                    root = ET.fromstring(ome_xml_str)

                    # Define the OME namespace (required for parsing)
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

                    # Find all <Channel> tags and get the Name attribute
                    channel_markers = [
                        channel.attrib['Name']
                        for channel in root.findall(".//ome:Channel", ns)
                        if 'Name' in channel.attrib
                    ]

                    self.generate_marker_txt(channel_markers)
                # catch any errors
                except Exception as e:
                    show_info(f"Error parsing OME metadata: {e}")
                    show_info("Notice! Your image file does not have valid OME metadata! Please include the markers manually.")

            elif str(first_image_path).endswith('.qptiff'):
                bftools_addr = os.path.join(
                    os.getcwd(),
                    "src/bftools/"
                )
                ome_meta_addr = os.path.join(
                    os.getcwd(),
                    "src/ome_metadata.txt"
                )
                result = subprocess.run(f"{bftools_addr}showinf -nopix -omexml {first_image_path} > {ome_meta_addr}", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    with open(f"{ome_meta_addr}", "r") as f:
                        lines = f.readlines()

                    # Extract biomarker names using regex
                    biomarkers = list()
                    for line in lines:
                        match = re.search(r"Biomarker\s+#\d+:\s+(.*)", line)
                        if match:
                            biomarkers.append(match.group(1).strip())

                    self.generate_marker_txt(biomarkers)

                else:
                    show_info("Notice! Your image file does not have valid OME metadata! Please include the markers manually.")

            else:
                show_info("Notice! We only support automatic extraction from .tiff, .tif, and .qptiff files. Please include the markers manually.")

    
        except Exception as e:
            print(f"CSV parsing error: {e}")
            show_info("Notice! Your CSV file has a parsing error!")
            return


    def generate_marker_txt(self, input_markers):

        automatic_marker_path = os.path.join(
            os.getcwd(),
            "src/AUTOMATIC_markers_batch.txt"
        )

        txt_to_write = ""
        for idx, marker in enumerate(input_markers):
            txt_to_write += f"{marker}\n"

        with open(automatic_marker_path, 'w') as f:

            f.write(txt_to_write)

        if os.path.exists(automatic_marker_path):
            self.params_panel.marker_file.value = automatic_marker_path
            self.add_marker()
        else:
            show_info("Notice! Automatic marker file creation failed! Please include the markers manually.")

    # parse the json file, if want to add more parameters, consider this function
    def parse_json(self):
        # to ensure whether the json file exists
        if not os.path.exists(self.params_panel.json_file.value):
            print("json file does not exist")
            show_info("Notice! Your json file does not exist!")
        else:
            print("json file exists")
            # try to parse the json file to update the hyper-parameters dict
            try:
                with open(self.params_panel.json_file.value, 'r') as f:
                    new_dict = json.load(f)
                    self.params_panel.device.value = new_dict['device']
                    self.params_panel.batch_size.value = new_dict['batch_size']
                    self.params_panel.main_dir.value = pathlib.Path(new_dict['main_dir'])
                    self.params_panel.strict.value = new_dict['strict']
                    self.params_panel.infer.value = new_dict['infer']
                    self.params_panel.min_cells.value = new_dict['min_cells']
                    self.params_panel.n_regions.value = new_dict['n_regions']
                    self.params_panel.normalize.value = new_dict['normalize']
                    self.params_panel.blur.value = new_dict['blur']
                    self.params_panel.upper_limit.value = new_dict['upper_limit']
                    self.params_panel.confidence.value = new_dict['confidence']
                    self.params_panel.cell_size.value = new_dict['cell_size']
                    self.params_panel.cell_type_confidence.value = new_dict['cell_type_confidence']
            except:
                print("json file parsing error")
                show_info("Notice! Your json file has a parsing error!")

    # NOTE: the marker file should be a .txt file
    def add_marker(self):
        marker_path = self.params_panel.marker_file.value

        # TODO: YOU CAN HAVE SOME FORMAT CHECKING HERE

        markers = list()
        # read the marker file line by line
        with open(marker_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    markers.append(line)

        new_lbl_txt = ""
        for idx, marker in enumerate(markers):
            new_lbl_txt += f"{idx}. {marker}, "
            # for the last element, do not have ,
            if idx == len(markers) - 1:
                new_lbl_txt = new_lbl_txt[:-2]
            if idx > 0 and idx % 4 == 0:
                new_lbl_txt += "\n"
        self.label_txt.setText(new_lbl_txt)

    def update_status_txt(self, txt):
        self.params_panel.job_status.value = txt

    def update_call_btn(self, status):
        if status:
            self.params_panel.call_button.text = "Set Parameters and Run Annotator"
            self.params_panel.call_button.show()
        else:
            self.params_panel.call_button.hide()


class GUIIntegrater(QWidget):
    def __init__(self):
        super().__init__()
        self.viewer = napari.current_viewer()
        self.params_panel.json_file.changed.connect(self.parse_json)

        # self.params_panel.image_file.changed.connect(self.add_image)
        self.params_panel.image_file.changed.connect(self.add_image_enhanced)
        self.params_panel.marker_file.changed.connect(self.add_marker)
        self.params_panel.mask_file.changed.connect(self.add_mask)

        self.params_panel.job_status.bind(self.update_status_txt)
        self.params_panel.call_button.bind(self.update_call_btn)

        self.viewer.layers.events.removed.connect(self.handle_delete_file)


        self.setLayout(QVBoxLayout())
        # set the title
        self.setWindowTitle("Single Process")

        self.label_txt = QLabel("No markers found.")

        self.cell_types_txt = QLabel(
            'Cell type names will be displayed here.'
        )
        self.intensity_txt = QLabel(
            "The intensity will be displayed here."
        )
        # self.layout().addWidget(self.label_txt)
        btn_main_panel = QPushButton("Multiplexed Image Annotator")
        btn_label_panel = QPushButton("Marker/Antibody Panel")
        btn_cell_types_panel = QPushButton("Cell type Panel")
        btn_intensity_panel = QPushButton("Cell-level Intensity")


        btn_main_panel.clicked.connect(self.launchMainPanel)
        btn_label_panel.clicked.connect(self.launchLabelPanel)
        btn_cell_types_panel.clicked.connect(self.launchCellTypesPanel)
        btn_intensity_panel.clicked.connect(self.launchIntensityPanel)
        
        self.layout().addWidget(btn_main_panel)
        self.layout().addWidget(btn_label_panel)
        self.layout().addWidget(btn_cell_types_panel)
        self.layout().addWidget(btn_intensity_panel)
        

        self.working_dir_addr = os.path.join(
            os.getcwd(),
            "src/multiplexed_image_annotator/cell_type_annotation/_working_dir_temp/"
        )

        self.hyper_params = dict()

        self.files_paths = ["", ""]
        self.files_idices = [-1, -1]

        # NOTE: Please check intensity dict here
        self.intensity_dict = dict()





    # after the segmentation algo is done, load the output_img from the temp folder
    def load_img(self):
        working_dir = pathlib.Path(self.working_dir_addr)
        output_addr = f"{working_dir}/output_img.png"
        output_addr_2 = f"{working_dir}/output_img_2.png"

        try: 
            output_img = imageio.imread(output_addr)
            self.viewer.add_labels(output_img, name="cell_type_map")
            if os.path.exists(output_addr_2):
                output_img_2 = imageio.imread(output_addr_2)
                self.viewer.add_labels(output_img_2, name="tissue_region")
        except:
            print("output_img reading error")
            show_info("Notice! Your output_img reading error! The process may have problem.")
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
        self.update_status_txt("You do not have running jobs.")
        self.update_call_btn(True)

        # TODO: if the algorithm returns a string, update your string with the function below
        # new_cell_types_txt = "We obtained some cell types from the algorithm."
        # self.update_cell_types_txt(new_cell_types_txt)

        show_info("Your job is done! The output image is loaded.")



    # the main function to launch the seg algo
    def seg_algo_caller(self):
        addr_working = pathlib.Path(self.working_dir_addr)
        if os.path.exists(addr_working):
            shutil.rmtree(addr_working)
        print(f"addr_working: {addr_working}")
        # os.mkdir(addr_working)
        os.mkdir(addr_working)
        with open(f"{addr_working}/hyperparams.json", 'w') as f:
            json.dump(self.hyper_params, f)
        self.update_status_txt("Your job is running, please wait for the result.")
        self.intensity_dict, cell_type_text = gui_api.gui_api(addr_working)
        self.update_cell_types_txt(cell_type_text)
        self.intensity_display = "Cell-level expression intensity has been detected.\nIt will be displayed here.\nPLEASE SELECT MASK LAYER TO USE THIS FUNCTION."
        self.intensity_txt.setText(
            self.intensity_display
        )


    # constructing the UI
    @magicgui(
        call_button="Set Parameters and Run Annotator",
        layout='vertical',
        device={'choices': ['cpu', 'cuda']},
        batch_size={'min': 1, 'max': 10000, 'step': 16},
        blur={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
        upper_limit={'widget_type': 'FloatSlider', 'min': 95, 'max': 100},
        cell_size={'min': 1, 'max': 100},
        min_cells={'min': -1, 'max': 10000},
        n_regions={'min': -1, 'max': 50},
        confidence={'widget_type': 'FloatSlider', 'min': 0, 'max': 1},
        main_dir={'widget_type': 'FileEdit', 'mode': 'd'},
        job_status={'widget_type': 'Label', 'value': 'You do not have running jobs.'}
    )
    def params_panel(
        self,
        json_file=pathlib.Path('PLEASE SELECT YOUR PARAMS JSON FILE (OPTIONAL)'),
        image_file=pathlib.Path('PLEASE SELECT YOUR IMAGE FILE (REQUIRED)'),
        marker_file=pathlib.Path('PLEASE SELECT YOUR MARKER FILE (REQUIRED)'),
        mask_file=pathlib.Path('PLEASE SELECT YOUR MASK FILE (REQUIRED)'),
        device='cuda',
        batch_size=128,
        main_dir=pathlib.Path('PLEASE SELECT YOUR MAIN DIR (REQUIRED)'),
        strict=False,
        infer=True,
        min_cells=50,
        n_regions=5,
        normalize=True,
        blur=0.4,
        upper_limit=99.8,
        confidence=0.3,
        cell_size=30,
        job_status='You do not have running jobs.',
        cell_type_confidence=None,
    ):
        new_dict = {
            'json_file': str(json_file),
            'image_file': str(image_file),
            'marker_file': str(marker_file),
            'mask_file': str(mask_file),
            'device': device,
            'batch_size': batch_size,
            'main_dir': str(main_dir),
            'strict': strict,
            'infer': infer,
            'min_cells': min_cells,
            'n_regions': n_regions,
            'normalize': normalize,
            'blur': blur,
            'upper_limit': upper_limit,
            'confidence': confidence,
            'cell_size': cell_size,
            'cell_type_confidence': cell_type_confidence,
            'n_jobs': 0
        }
        self.hyper_params = new_dict
        print(pathlib.Path('.'))
        is_valid = True
        # check whether the image and marker files exist
        if not os.path.exists(image_file):
            print("image file does not exist")
            show_info("Notice! Your image file does not exist!")
            is_valid = False


        if not os.path.exists(marker_file):
            print("marker file does not exist")
            show_info("Notice! Your marker file does not exist!")
            is_valid = False

        # check whether marker file is a txt file
        if not str(marker_file.suffix) == '.txt':
            print("marker file is not a txt file")
            show_info("Notice! Your marker file is not in right format (.txt expected)!")
            is_valid = False

        if not os.path.exists(mask_file):
            print("mask file does not exist")
            show_info("Notice! Your mask file does not exist!")
            is_valid = False

        # check whether the main_dir exists and is a directory
        if not os.path.exists(main_dir):
            print("main_dir does not exist")
            show_info("Notice! Your main_dir does not exist!")
            is_valid = False
        elif not os.path.isdir(main_dir):
            print("main_dir is not a directory")
            show_info("Notice! Your main_dir is not a directory!")
            is_valid = False

        if is_valid:
            # start the job
            show_info("Your job has been submitted!")
            self.update_status_txt("Your job has been submitted!")
            self.update_call_btn(False)
            worker_seg_algo = napari.qt.threading.create_worker(
                self.seg_algo_caller,
            )

            worker_seg_algo.start()
            
            worker_seg_algo.finished.connect(self.load_img)

        else:
            self.update_status_txt("Please pay attention to your params!")


    # parse the json file, if want to add more parameters, consider this function
    def parse_json(self):
        # to ensure whether the json file exists
        if not os.path.exists(self.params_panel.json_file.value):
            print("json file does not exist")
            show_info("Notice! Your json file does not exist!")
        else:
            print("json file exists")
            # try to parse the json file to update the hyper-parameters dict
            try:
                with open(self.params_panel.json_file.value, 'r') as f:
                    new_dict = json.load(f)
                    self.params_panel.device.value = new_dict['device']
                    self.params_panel.batch_size.value = new_dict['batch_size']
                    self.params_panel.main_dir.value = pathlib.Path(new_dict['main_dir'])
                    self.params_panel.strict.value = new_dict['strict']
                    self.params_panel.infer.value = new_dict['infer']
                    self.params_panel.min_cells.value = new_dict['min_cells']
                    self.params_panel.n_regions.value = new_dict['n_regions']
                    self.params_panel.normalize.value = new_dict['normalize']
                    self.params_panel.blur.value = new_dict['blur']
                    self.params_panel.upper_limit.value = new_dict['upper_limit']
                    self.params_panel.confidence.value = new_dict['confidence']
                    self.params_panel.cell_size.value = new_dict['cell_size']
                    self.params_panel.cell_type_confidence.value = new_dict['cell_type_confidence']
            except:
                print("json file parsing error")
                show_info("Notice! Your json file has a parsing error!")


    def add_image(self):
        img_path = self.params_panel.image_file.value
        if not os.path.exists(img_path):
            return

        try:
            img_name = "multiplexed_image"
            img = imread(img_path)
            if self.files_idices[0] != -1:
                self.viewer.layers[img_name].data = img
            else:
                user_img = self.viewer.add_image(
                    img,
                    name=img_name
                )
                self.files_idices[0] = 1
            
        except:
            print("image reading error")
            show_info("Notice! Input image reading error!")

    def add_image_enhanced(self):
        img_path = self.params_panel.image_file.value
        if not os.path.exists(img_path):
            return

        try:
            img_name = "multiplexed_image"
            img = imread(img_path)
            if self.files_idices[0] != -1:
                self.viewer.layers[img_name].data = img
            else:
                user_img = self.viewer.add_image(
                    img,
                    name=img_name
                )
                self.files_idices[0] = 1
            
        except Exception as e:
            print(f"image reading error as: \n{e}")
            show_info("Notice! Input image reading error!")
            return

        if str(img_path).endswith('.tiff') or str(img_path).endswith('.tif'):
            try:
                with tifffile.TiffFile(img_path) as tif:
                    # print("OME metadata:", tif.ome_metadata)
                    ome_xml_str = tif.ome_metadata
                # Parse the XML
                root = ET.fromstring(ome_xml_str)

                # Define the OME namespace (required for parsing)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

                # Find all <Channel> tags and get the Name attribute
                channel_markers = [
                    channel.attrib['Name']
                    for channel in root.findall(".//ome:Channel", ns)
                    if 'Name' in channel.attrib
                ]

                # print("Marker names:", channel_markers)
                self.generate_marker_txt(channel_markers)
            # catch any errors
            except Exception as e:
                show_info(f"Error parsing OME metadata: {e}")
                show_info("Notice! Your image file does not have valid OME metadata! Please include the markers manually.")

        elif str(img_path).endswith('.qptiff'):
            bftools_addr = os.path.join(
                os.getcwd(),
                "src/bftools/"
            )
            ome_meta_addr = os.path.join(
                os.getcwd(),
                "src/ome_metadata.txt"
            )
            result = subprocess.run(f"{bftools_addr}showinf -nopix -omexml {img_path} > {ome_meta_addr}", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                with open(f"{ome_meta_addr}", "r") as f:
                    lines = f.readlines()

                # Extract biomarker names using regex
                biomarkers = list()
                for line in lines:
                    match = re.search(r"Biomarker\s+#\d+:\s+(.*)", line)
                    if match:
                        biomarkers.append(match.group(1).strip())

                self.generate_marker_txt(biomarkers)

            else:
                show_info("Notice! Your image file does not have valid OME metadata! Please include the markers manually.")

        else:
            show_info("Notice! We only support automatic extraction from .tiff, .tif, and .qptiff files. Please include the markers manually.")

    def generate_marker_txt(self, input_markers):

        automatic_marker_path = os.path.join(
            os.getcwd(),
            "src/AUTOMATIC_markers_single.txt"
        )

        txt_to_write = ""
        for idx, marker in enumerate(input_markers):
            txt_to_write += f"{marker}\n"

        with open(automatic_marker_path, 'w') as f:

            f.write(txt_to_write)

        if os.path.exists(automatic_marker_path):
            self.params_panel.marker_file.value = automatic_marker_path
            self.add_marker()
        else:
            show_info("Notice! Automatic marker file creation failed! Please include the markers manually.")


    # NOTE: the marker file should be a .txt file
    def add_marker(self):
        marker_path = self.params_panel.marker_file.value
        if not os.path.exists(marker_path):
            return
        try:
            self.markers = list()
            # read the marker file line by line
            with open(marker_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        self.markers.append(line)

            new_lbl_txt = ""
            for idx, marker in enumerate(self.markers):
                new_lbl_txt += f"{idx}. {marker}, "
                # for the last element, do not have ,
                if idx == len(self.markers) - 1:
                    new_lbl_txt = new_lbl_txt[:-2]
                if idx > 0 and idx % 5 == 0:
                    new_lbl_txt += "\n"

            self.label_txt.setText(new_lbl_txt)
        except:
            print("marker reading error")
            show_info("Notice! Input marker reading error!")


    def add_mask(self):
        mask_path = self.params_panel.mask_file.value
        if not os.path.exists(mask_path):
            return
        try:
            mask_name = "cell_mask"
            mask = imageio.imread(mask_path)
            if self.files_idices[1] != -1:
                self.viewer.layers[mask_name].data = mask
            else:
                user_mask = self.viewer.add_image(
                    mask,
                    name=mask_name
                )
                self.files_idices[1] = 1
                @user_mask.mouse_drag_callbacks.append

                def sync_label_name(layer, event):
                    if len(self.intensity_dict) == 0:
                        return
                    data_coordinates = layer.world_to_data(event.position)
                    val = layer.get_value(data_coordinates)
                    intensity = self.intensity_dict.get(val)
                    new_txt = ""
                    if intensity is not None:
                        for idx, marker in enumerate(self.markers):
                            new_txt += f"{marker}: {intensity[idx]:{1}.{4}}, "
                            # for the last element, do not have ,
                            if idx == len(self.markers) - 1:
                                new_txt = new_txt[:-2]
                            if idx > 0 and idx % 5 == 0:
                                new_txt += "\n"
                        self.intensity_txt.setText(new_txt)
                    else:
                        new_txt = "Cell-level expression intensity has been detected.\nIt will be displayed here.\nPLEASE SELECT MASK LAYER TO USE THIS FUNCTION."
                        self.intensity_txt.setText(new_txt)

            self.intensity_txt.setText(
                    self.intensity_display
            )

        except:
            print("mask reading error")
            show_info("Notice! Input mask reading error!")

    def update_status_txt(self, txt):
        self.params_panel.job_status.value = txt

    def update_call_btn(self, status):
        if status:
            self.params_panel.call_button.text = "Set Parameters and Run Annotator"
            self.params_panel.call_button.show()
        else:
            self.params_panel.call_button.hide()

    def update_cell_types_txt(self, txt):
        self.cell_types_txt.setText(txt)

    def handle_delete_file(self, event):

        if event.value.name == "multiplexed_image":
            self.files_idices[0] = -1
            self.params_panel.image_file.value = "PLEASE SELECT YOUR IMAGE FILE (REQUIRED)"

        if event.value.name == "cell_mask":
            self.files_idices[1] = -1
            self.params_panel.mask_file.value = "PLEASE SELECT YOUR MASK FILE (REQUIRED)"

                


    def launchLabelPanel(self):
        self.viewer.window.add_dock_widget(
            self.label_txt,
            name="Marker/Antibody Panel"
        )

    def launchIntensityPanel(self):
        self.viewer.window.add_dock_widget(
            self.intensity_txt,
            name="Cell-level average intensity"
        )

    def launchCellTypesPanel(self):
        self.viewer.window.add_dock_widget(
            self.cell_types_txt,
            name="Cell types"
        )

    def launchMainPanel(self):
        self.viewer.window.add_dock_widget(
            self.params_panel,
            name="Multiplexed Image Annotator",
        )
