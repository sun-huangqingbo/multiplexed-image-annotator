import os
import gdown

if not os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models"):
    os.makedirs("src/multiplexed_image_annotator/cell_type_annotation/models")

immune_base_url = "https://drive.google.com/file/d/1EOe216MIV7o_pxAHIYg0KUab94BnhP0Z/view?usp=sharing"
immune_base_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_base.pth")
immune_extended_url = "https://drive.google.com/file/d/1U8X-ka5hk3MvVUZ6nO7Nx30WrT5Rc1yU/view?usp=sharing"
immune_extended_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_extended.pth")
immune_full_url = "https://drive.google.com/file/d/1-KCwsysGks8BUXElAoF2rAqUvBpD4mtB/view?usp=sharing"
immune_full_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_full.pth")
structure_url = "https://drive.google.com/file/d/1fuC78XYHSs0acN0m9b25xN86ujBxE6A5/view?usp=sharing"
structure_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "struct.pth")
nerve_url = "https://drive.google.com/file/d/1B8hv3xG5kI0TsSY5XLwAyHI6D7KKQ8y1/view?usp=sharing"
nerve_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "nerve.pth")

immune_base_impute_url = "https://drive.google.com/file/d/1034ex__rMmMz7w8PPCY1e8YHWwl62J9H/view?usp=sharing"
immune_base_impute_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_base_impute.pth")
immune_extended_impute_url = "https://drive.google.com/file/d/1iQUY3JUueCNTducSNUmijzQVIkaTuJ0N/view?usp=sharing"
immune_extended_impute_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_extended_impute.pth")
immune_full_impute_url = "https://drive.google.com/file/d/1vK9JgVlO_PtcI_B9mPXCp6eamff0S2rD/view?usp=sharing"
immune_full_impute_path = os.path.join(f"src/multiplexed_image_annotator/cell_type_annotation/models", "immune_full_impute.pth")



# Download the file from `url` and save it
gdown.download(immune_base_url, immune_base_path, quiet=False, fuzzy=True)
gdown.download(immune_extended_url, immune_extended_path, quiet=False, fuzzy=True)
gdown.download(immune_full_url, immune_full_path, quiet=False, fuzzy=True)
gdown.download(structure_url, structure_path, quiet=False, fuzzy=True)
gdown.download(nerve_url, nerve_path, quiet=False, fuzzy=True)


gdown.download(immune_base_impute_url, immune_base_impute_path, quiet=False, fuzzy=True)
gdown.download(immune_extended_impute_url, immune_extended_impute_path, quiet=False, fuzzy=True)
gdown.download(immune_full_impute_url, immune_full_impute_path, quiet=False, fuzzy=True)