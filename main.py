import os
import numpy as np
import pandas as pd

from multiplexed_image_annotator.cell_type_annotation.model import Annotator
import argparse


def run(marker_list_path, image_path, mask_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence):

    # write image and mask paths to a csv file
    temp = [[image_path, mask_path]]
    pd.DataFrame(temp).to_csv(os.path.join(main_dir, "images.csv"), index=False, header=["image_path", "mask_path"])
    
    path_ = os.path.join(main_dir, "images.csv")
    annotator = Annotator(marker_list_path, path_, device, main_dir, batch_id, strict, infer, min_cells, normalize, blur, amax, confidence, cell_size, cell_type_confidence)
    if not annotator.channel_parser.immune_base and not annotator.channel_parser.immune_extended and not annotator.channel_parser.immune_full and not annotator.channel_parser.struct and not annotator.channel_parser.nerve:
        raise ValueError("No panels are applied. Please check the marker list.")
    annotator.preprocess()
    annotator.predict(bs)
    annotator.generate_heatmap(integrate=True)
    annotator.export_annotations()
    annotator.tissue_region_analysis(n_regions)
    annotator.colorize(from_script=True)
    annotator.cell_type_composition()
    annotator.clear_tmp()

    intensity_dict = {}
    for i in range(len(annotator.preprocessor.intensity_full[0])):
        intensity_dict[i + 1] = annotator.preprocessor.intensity_full[0][i]
    intensity_dict[0] = np.zeros_like(annotator.preprocessor.intensity_full[0][0])
    names = annotator.get_cell_type_names()

    return intensity_dict, names
    

def batch_run(marker_list_path, image_path, device, main_dir, batch_id, bs, strict, infer, min_cells, n_regions, normalize, blur, amax, confidence, cell_size, cell_type_confidence):
    annotator = Annotator(marker_list_path, image_path, device, main_dir, batch_id, strict, infer, min_cells, normalize, blur, amax, confidence, cell_size, cell_type_confidence)
    if not annotator.channel_parser.immune_base and not annotator.channel_parser.immune_extended and not annotator.channel_parser.immune_full and not annotator.channel_parser.struct and not annotator.channel_parser.nerve:
        raise ValueError("No panels are applied. Please check the marker list.")
    annotator.preprocess()
    annotator.predict(bs)
    annotator.generate_heatmap(integrate=True)
    annotator.export_annotations()
    annotator.tissue_region_analysis(n_regions)
    annotator.colorize(from_script=True)
    annotator.cell_type_composition()
    annotator.clear_tmp()
    


def parse_args():
    parser = argparse.ArgumentParser(description='Process images with markers')
    
    # Required arguments
    parser.add_argument('--marker-list-path', type=str, required=True,
                      help='Path to the markers text file')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on (cuda/cpu)')
    parser.add_argument('--main-dir', type=str, default='./',
                      help='Main directory path')
    parser.add_argument('--batch-id', type=str, required=True,
                      help='Batch identifier')
    
    # Optional arguments with defaults
    parser.add_argument('--strict', action='store_true',
                      help='Enable strict mode')
    parser.add_argument('--infer', action='store_true', default=True,
                      help='Enable inference')
    parser.add_argument('--min-cells', type=int, default=-1,
                      help='Minimum number of cells')
    parser.add_argument('--n-regions', type=int, default=3,
                      help='Number of regions')
    parser.add_argument('--normalize', action='store_true', default=True,
                      help='Enable normalization')
    parser.add_argument('--blur', type=float, default=0.3,
                      help='Blur factor')
    parser.add_argument('--amax', type=float, default=99.8,
                      help='Maximum amplitude')
    parser.add_argument('--confidence', type=float, default=0.3,
                      help='Confidence threshold')
    parser.add_argument('--cell-type-confidence', type=float, default=None,
                      help='Cell type confidence threshold')
    parser.add_argument('--bs', type=int, default=128,
                      help='Batch size')
    parser.add_argument('--cell-size', type=int, default=30,
                      help='Cell size')
    
    # Mode selection (single or batch)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image-path', type=str,
                      help='Path to single image file')
    group.add_argument('--batch-csv', type=str,
                      help='Path to CSV file for batch processing')
    
    # Required for single mode only
    parser.add_argument('--mask-path', type=str,
                      help='Path to mask file (required for single image mode)')
    
    args = parser.parse_args()
    
    # Validate that mask_path is provided when in single image mode
    if args.image_path and not args.mask_path:
        parser.error("--mask-path is required when using --image-path")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if args.batch_csv:
        batch_run(
            marker_list_path=args.marker_list_path,
            image_path=args.batch_csv,
            device=args.device,
            main_dir=args.main_dir,
            batch_id=args.batch_id,
            bs=args.bs,
            strict=args.strict,
            infer=args.infer,
            min_cells=args.min_cells,
            n_regions=args.n_regions,
            normalize=args.normalize,
            blur=args.blur,
            amax=args.amax,
            confidence=args.confidence,
            cell_size=args.cell_size,
            cell_type_confidence=args.cell_type_confidence
        )
    else:
        run(
            marker_list_path=args.marker_list_path,
            image_path=args.image_path,
            mask_path=args.mask_path,
            device=args.device,
            main_dir=args.main_dir,
            batch_id=args.batch_id,
            bs=args.bs,
            strict=args.strict,
            infer=args.infer,
            min_cells=args.min_cells,
            n_regions=args.n_regions,
            normalize=args.normalize,
            blur=args.blur,
            amax=args.amax,
            confidence=args.confidence,
            cell_size=args.cell_size,
            cell_type_confidence=args.cell_type_confidence
        )