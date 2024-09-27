from .model import Annotator
import torch
from .utils import gui_run

marker_list_path = r"C:\Users\57539\Downloads\for_paul\for_paul\markers.txt"
image_path = "data/debug.csv" # r"E:\HPA_TMA\20240426_HPA_TMA_G228_normal_paths.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_dir = "./"
batch_id = ""
strict = False
normalization = True
blur = 0.5
confidence = 0.25


# image_path = r"E:\HPA_TMA\20240426_HPA_TMA_G228_normal_114\20240426_HPA_TMA_cropped_32.tif"
# mask_path = r"E:\HPA_TMA\20240426_HPA_TMA_G228_normal_114\20240426_HPA_TMA_cropped_32_cp_masks.png"

# img = gui_run(marker_list_path, image_path, mask_path, device, main_dir, batch_id, strict, normalization, blur, confidence, None)


annotator = Annotator(marker_list_path, image_path, device, main_dir, batch_id, strict, normalization, blur, confidence)
annotator.preprocess()
annotator.predict(128)
annotator.generate_heatmap(integrate=True)
# annotator.umap_visualization()
annotator.export_annotations()
annotator.colorize()
annotator.cell_type_composition()
annotator.neighborhood_analysis(n_neighbors=15, integrate=True, normalize=False)
annotator.clear_tmp()

intensity_dict = {}
for i in range(len(annotator.preprocessor.intensity_full[0])):
    intensity_dict[i + 1] = annotator.preprocessor.intensity_full[0][i]
intensity_dict[0] = np.zeros_like(annotator.preprocessor.intensity_full[0][0])

print("Done")