import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from functools import partial

import torch.nn as nn

import timm.models.vision_transformer

from .markerParse import MarkerParser
from .preprocess import ImageProcessor
from .logger import Logger
from .utils import *
from .spatial_methods import _tissue_region_partition

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, HDBSCAN

import umap

import pickle



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome
    
def vit_s(**kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=288, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_l(**kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=288 * 2, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
def vit_m(**kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model    

def vit_tiny(**kwargs):
    model = VisionTransformer(
        patch_size=4, embed_dim=144, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model    

class Annotator(object):
    """
    Annotator class to predict cell types and tissue structures using the provided models
    """
    def __init__(self, marker_list_path, image_path, device, main_dir = './', batch_id='', strict=True, infer=True, min_cells=-1, normalize=True, blur=False, amax=1, confidence=0.25, cell_size = 30, cell_type_confidence=None):
        self.device = device
        self.cell_types = ["B cell", "CD4 T cell", "CD8 T cell", "Dendritic cell", "Regulatory T cell", "Granulocyte cell", 
                           "Mast cell", "M1 macrophage cell", "M2 macrophage cell", "Natural killer cell", "Plasma cell",
                           "Endothelial cell", "Epithelial cell", "Stroma cell", "Smooth muscle", "Proliferating/tumor cell", "Nerve cell", "Others"]


        self.batch_id = batch_id

        self.logger = Logger(main_dir)
        # hyperparameters as a dictionary
        hyperparameters = {
            "Batch name": batch_id,
            "Strictly match panel(s)": strict,
            "Normalize image(s)": normalize,
            "Image blurring kernel size": blur,
            "Percentile of intensity to upper clip": amax,
            "Confidence threshold": confidence,
            "Estimated cell size (in pixels)": cell_size
        }

        self.logger.log_all_hyperparameters(hyperparameters)
        self.logger.log("")
        self.logger.log("Start parsing the marker list.")


        self.channel_parser = MarkerParser(strict=strict, logger=self.logger)

        self.channel_parser.parse(marker_list_path)

        self.preprocessor = ImageProcessor(image_path, self.channel_parser, main_dir, device, batch_id, infer, normalize, blur, amax, cell_size, self.logger)
        self._loaded = False

        self._n_images = 0
        self.min_cells = min_cells

        self.annotations = []
        self.confidence = []

        self.immune_annotations = []
        self.struct_annotations = []
        self.nerve_annotations = []

        self.immune_base_pred = []
        self.immune_extended_pred = []
        self.immune_full_pred = []
        self.struct_pred = []
        self.nerve_pred = []

        self.immune_base_pred_II = []
        self.immune_extended_pred_II = []
        self.immune_full_pred_II = []
        self.struct_pred_II = []
        self.nerve_pred_II = []

        self.confidence_thresh = confidence

        self.extra_cell_types = self.min_cells > 0 

        self.temp_dir = os.path.join(main_dir, "tmp")
        self.result_dir = os.path.join(main_dir, "results")
        
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        if cell_type_confidence is None:
            self.cell_type_confidence = {'B cell': -1, 'CD4 T cell': -1, 'CD8 T cell': -1, 'Dendritic cell': -1, 'Regulatory T cell': -1, 'Granulocyte cell': -1, 'Mast cell': -1, 
                        'M1 macrophage cell': -1, 'M2 macrophage cell': -1, 'Natural killer cell': -1, 'Plasma cell': -1, 'Endothelial cell': -1,
                        'Epithelial cell': -1, 'Stroma cell': -1, 'Smooth muscle': -1, 'Proliferating/tumor cell': -1, 'Nerve cell': -1, 'Others': -1}
        else:
            self.cell_type_confidence = cell_type_confidence

    def preprocess(self):
        self.preprocessor.transform()
        self._n_images = self.preprocessor._n_images

    def clear(self):
        self.immune_base_pred = []
        self.immune_extended_pred = []
        self.immune_full_pred = []
        self.struct_pred = []
        self.nerve_pred = []

        self.immune_base_pred_II = []
        self.immune_extended_pred_II = []
        self.immune_full_pred_II = []
        self.struct_pred_II = []
        self.nerve_pred_II = []

        self.annotations = []
    
    def load_models(self):
        if os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models/immune_base.pth"):
            self.immune_base_model = vit_s(img_size=40, in_chans=7, num_classes=5, drop_path_rate=0.1, global_pool=False)
            checkpoint = torch.load("src/multiplexed_image_annotator/cell_type_annotation/models/immune_base.pth", map_location=self.device)["model"]
            self.immune_base_model.load_state_dict(checkpoint)
            self.immune_base_model.eval()
            self.immune_base_model.to(self.device)
        else:
            print("Immune base model not found")
            self.logger.log("Immune base model not found")
        
        if os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models/immune_extended.pth"):
            self.immune_extended_model = vit_m(img_size=40, in_chans=10, num_classes=8, drop_path_rate=0.1, global_pool=False)
            checkpoint = torch.load("src/multiplexed_image_annotator/cell_type_annotation/models/immune_extended.pth", map_location=self.device)["model"]
            self.immune_extended_model.load_state_dict(checkpoint)
            self.immune_extended_model.eval()
            self.immune_extended_model.to(self.device)
        else:
            print("Immune extended model not found")
            self.logger.log("Immune extended model not found")
        
        if os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models/immune_full.pth"):
            self.immune_full_model = vit_l(img_size=40, in_chans=15, num_classes=12, drop_path_rate=0.1, global_pool=False)
            checkpoint = torch.load("src/multiplexed_image_annotator/cell_type_annotation/models/immune_full.pth", map_location=self.device)["model"]
            self.immune_full_model.load_state_dict(checkpoint)
            self.immune_full_model.eval()
            self.immune_full_model.to(self.device)
        else:
            print("Immune full model not found")
            self.logger.log("Immune full model not found")
        
        if os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models/struct.pth"):
            self.struct_model = vit_s(img_size=40, in_chans=7, num_classes=6, drop_path_rate=0.1, global_pool=False)
            checkpoint = torch.load("src/multiplexed_image_annotator/cell_type_annotation/models/struct.pth", map_location=self.device)["model"]
            self.struct_model.load_state_dict(checkpoint)
            self.struct_model.eval()
            self.struct_model.to(self.device)
        else:
            print("Tissue structure model not found")
            self.logger.log("Tissue structure model not found")

        if os.path.exists("src/multiplexed_image_annotator/cell_type_annotation/models/nerve.pth"):
            self.nerve_model = vit_tiny(img_size=40, in_chans=3, num_classes=2, drop_path_rate=0.1, global_pool=False)
            checkpoint = torch.load("src/multiplexed_image_annotator/cell_type_annotation/models/nerve.pth", map_location=self.device)["model"]
            self.nerve_model.load_state_dict(checkpoint)
            self.nerve_model.eval()
            self.nerve_model.to(self.device)
        else:
            print("Nerve cell model not found")
            self.logger.log("Nerve cell model not found")
        
        self._loaded = True
        
        
    def predict(self, batch_size=32):
        self.logger.log("\nStart predicting cell types and tissue structures.")
        # check if models are loaded
        if self._loaded == False:
            self.load_models()

        # load pre-saved tensor
        for ii in range(self._n_images):
            if self.channel_parser.immune_full:
                f = os.path.join(self.temp_dir, f"{self.batch_id}_{ii}_immune_full.pt")
                if os.path.exists(f):
                    images = torch.load(f)
                else:
                    raise ValueError(f"Image tensor {f} not found")
                # predict immune full using batches
                temp = []
                for i in range(0, len(images), batch_size):
                    x = images[i:i+batch_size]
                    x = x.to(self.device, dtype = torch.float32, non_blocking=True)
                    pred_ = self.immune_full_model(x)
                    # softmax
                    pred_ = nn.functional.softmax(pred_, dim=1)
                    temp.append(pred_.detach().cpu().numpy())

                temp = np.concatenate(temp, axis=0)

                celltype_dict = {0: "CD4 T cell", 1: "CD8 T cell", 2: "Dendritic cell", 3: "B cell", 4: "M1 macrophage cell", 5: "M2 macrophage cell", 
                                 6: "Regulatory T cell", 7: "Granulocyte cell", 8: "Plasma cell", 9: "Natural killer cell", 10: "Mast cell", 11: "Others"}
                
                for c in celltype_dict.values():
                    if c not in self.applied_cell_types and c != "Others":
                        self.applied_cell_types.append(c)

                pred = []

                for j in range(len(temp)):
                    pred.append({celltype_dict[i]: temp[j][i] for i in range(len(temp[j]))})
                self.immune_full_pred.append(pred)

                del temp

                self.immune_annotations.append(pred)


            elif self.channel_parser.immune_extended:
                f = os.path.join(self.temp_dir, f"{self.batch_id}_{ii}_immune_extended.pt")
                if os.path.exists(f):
                    images = torch.load(f)
                else:
                    raise ValueError(f"Image tensor {f} not found")

                temp = []
                for i in range(0, len(images), batch_size):
                    x = images[i:i+batch_size]
                    x = x.to(self.device, dtype = torch.float32, non_blocking=True)
                    pred_ = self.immune_extended_model(x)
                    # softmax
                    pred_ = nn.functional.softmax(pred_, dim=1)
                    temp.append(pred_.detach().cpu().numpy())
                    
                temp = np.concatenate(temp, axis=0)

                celltype_dict = {0: "CD4 T cell", 1: "CD8 T cell", 2: "Dendritic cell", 3: "B cell", 4: "M1 macrophage cell",
                                    5: "M2 macrophage cell", 6: "Natural killer cell", 7: "Others"}
                
                
                pred = []
                for j in range(len(temp)):
                    pred.append({celltype_dict[i]: temp[j][i] for i in range(len(temp[j]))})
                self.immune_extended_pred.append(pred)

                del temp

                self.immune_annotations.append(pred)
        
            elif self.channel_parser.immune_base:
                f = os.path.join(self.temp_dir, f"{self.batch_id}_{ii}_immune_base.pt")
                if os.path.exists(f):
                    images = torch.load(f)
                else:
                    raise ValueError(f"Image tensor {f} not found")
                # predict immune base using batches
                temp = []
                for i in range(0, len(images), batch_size):
                    x = images[i:i+batch_size]
                    x = x.to(self.device, dtype = torch.float32, non_blocking=True)
                    pred_ = self.immune_base_model(x)
                    # softmax
                    pred_ = nn.functional.softmax(pred_, dim=1)
                    temp.append(pred_.detach().cpu().numpy())
                    # append the second highest prediction

                temp = np.concatenate(temp, axis=0)

                celltype_dict = {0: "B cell", 1: "CD4 T cell", 2: "CD8 T cell", 3: "Others", 4: "Dendritic cell"}

                for c in celltype_dict.values():
                    if c not in self.applied_cell_types and c != "Others":
                        self.applied_cell_types.append(c)

                pred = []

                for j in range(len(temp)):
                    pred.append({celltype_dict[i]: temp[j][i] for i in range(len(temp[j]))})

                self.immune_base_pred.append(pred)

                del temp

                self.immune_annotations.append(pred)
            else:
                print("No immune cell model to predict")
                self.logger.log("No immune cell model to predict")


            if self.channel_parser.struct:
                f = os.path.join(self.temp_dir, f"{self.batch_id}_{ii}_structure.pt")
                if os.path.exists(f):
                    images = torch.load(f)
                else:
                    raise ValueError(f"Image tensor {f} not found")
                # predict structure using batches
                temp = []
                for i in range(0, len(images), batch_size):
                    x = images[i:i+batch_size]
                    x = x.to(self.device, dtype = torch.float32, non_blocking=True)
                    pred_ = self.struct_model(x)
                    # softmax
                    pred_ = nn.functional.softmax(pred_, dim=1)
                    temp.append(pred_.detach().cpu().numpy())

                temp = np.concatenate(temp, axis=0)
  

                celltype_dict = {0: "Stroma cell" , 1: "Smooth muscle", 2: "Endothelial cell", 3: "Epithelial cell", 4: "Proliferating/tumor cell", 5: "Others"}

                pred = []

                for j in range(len(temp)):
                    pred.append({celltype_dict[i]: temp[j][i] for i in range(len(temp[j]))})
                self.struct_pred.append(pred)

                del temp

                self.struct_annotations.append(pred)

            else:
                print("No structure model to predict")
                self.logger.log("No structure model to predict")

            if self.channel_parser.nerve:
                f = os.path.join(self.temp_dir, f"{self.batch_id}_{ii}_nerve_cell.pt")
                if os.path.exists(f):
                    images = torch.load(f)
                else:
                    raise ValueError(f"Image tensor {f} not found")
                # predict nerve using batches
                temp = []
                for i in range(0, len(images), batch_size):
                    x = images[i:i+batch_size]
                    x = x.to(self.device, dtype = torch.float32, non_blocking=True)
                    pred_ = self.nerve_model(x)
                    # softmax
                    pred_ = nn.functional.softmax(pred_, dim=1)
                    temp.append(pred_.detach().cpu().numpy())

                temp = np.concatenate(temp, axis=0)
  

                celltype_dict = {0: "Nerve cell", 1: "Others"}

                pred = []

                for j in range(len(temp)):
                    pred.append({celltype_dict[i]: temp[j][i] for i in range(len(temp[j]))})
                self.nerve_pred.append(pred)

                del temp

                self.nerve_annotations.append(pred)

            else:
                print("No nerve cell model to predict")
                self.logger.log("No nerve cell model to predict")

        self.merge_by_voting()

        self.cell_types = self._get_unique_cell_types()
        # move Others to the end
        self.cell_types = np.delete(self.cell_types, np.where(self.cell_types == "Others"))
        self.cell_types = np.append(self.cell_types, "Others")
        self.colors = get_colors(len(self.cell_types))
        # save legend
        colors = {f"{self.cell_types[i]}": rgb_to_hex(self.colors[i]) for i in range(len(self.cell_types))}
        color_legend(self.result_dir, colors)
                

    def merge_by_voting(self):

        # full
        if len(self.immune_full_pred) > 0 and len(self.struct_pred) > 0 and len(self.nerve_pred) > 0:
            for i in range(len(self.immune_full_pred)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.immune_full_pred[i])):
                    vote = get_void_vote()
                    pred = self.immune_full_pred[i][j]
                    for k in pred:
                        vote[k] += pred[k]
                    o1 = pred["Others"]
                    pred = self.struct_pred[i][j]
                    for k in pred:
                        vote[k] += pred[k]
                    o2 = pred["Others"]
                    pred = self.nerve_pred[i][j]
                    for k in pred:
                        vote[k] += pred[k]
                    o3 = pred["Others"]
                    
                    max_vote = max(vote, key=vote.get)

                    thresh = min(o1, o2, o3, self.confidence_thresh) if self.cell_type_confidence[max_vote] < 0 else self.cell_type_confidence[max_vote]
                    if vote[max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(vote[max_vote]))

        elif len(self.immune_annotations) > 0 and len(self.struct_annotations) > 0:
            for i in range(len(self.immune_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.immune_annotations[i])):
                    vote = get_void_vote()
                    pred = self.immune_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o1 = pred["Others"]
                    pred = self.struct_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o2 = pred["Others"]
                    
                    max_vote = max(vote, key=vote.get)

                    thresh = min(o1, o2, self.confidence_thresh) if self.cell_type_confidence[max_vote] < 0 else self.cell_type_confidence[max_vote]
                    if vote[max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(vote[max_vote]))
        
        elif len(self.struct_annotations) > 0 and len(self.nerve_annotations) > 0:
            for i in range(len(self.struct_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.struct_annotations[i])):
                    vote = get_void_vote()
                    pred = self.struct_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o1 = pred["Others"]
                    pred = self.nerve_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o2 = pred["Others"]
                    
                    max_vote = max(vote, key=vote.get)

                    thresh = min(o1, o2, self.confidence_thresh) if self.cell_type_confidence[max_vote] < 0 else self.cell_type_confidence[max_vote]
                    if vote[max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(vote[max_vote]))

        elif len(self.immune_annotations) > 0 and len(self.nerve_annotations) > 0:
            for i in range(len(self.immune_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.immune_annotations[i])):
                    vote = get_void_vote()
                    pred = self.immune_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o1 = pred["Others"]
                    pred = self.nerve_annotations[i][j]
                    for k in pred:
                        if k != "Others":
                            vote[k] += pred[k]
                    o2 = pred["Others"]
                    
                    max_vote = max(vote, key=vote.get)

                    thresh = min(o1, o2, self.confidence_thresh) if self.cell_type_confidence[max_vote] < 0 else self.cell_type_confidence[max_vote]
                    if vote[max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(vote[max_vote]))

        elif len(self.immune_annotations) > 0:
            for i in range(len(self.immune_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.immune_annotations[i])):
                    max_vote = max(self.immune_annotations[i][j], key=self.immune_annotations[i][j].get)
                    thresh = self.cell_type_confidence[max_vote] if self.cell_type_confidence[max_vote] > 0 else self.confidence_thresh
                    if max_vote != "Others" and self.immune_annotations[i][j][max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(self.immune_annotations[i][j][max_vote]))
                    
        elif len(self.struct_annotations) > 0:
            for i in range(len(self.struct_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.struct_annotations[i])):
                    max_vote = max(self.struct_annotations[i][j], key=self.struct_annotations[i][j].get)
                    thresh = self.cell_type_confidence[max_vote] if self.cell_type_confidence[max_vote] > 0 else self.confidence_thresh
                    if max_vote != "Others" and self.struct_annotations[i][j][max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(self.struct_annotations[i][j][max_vote]))

        elif len(self.nerve_annotations) > 0:
            for i in range(len(self.nerve_annotations)):
                self.annotations.append([])
                self.confidence.append([])
                for j in range(len(self.nerve_annotations[i])):
                    max_vote = max(self.nerve_annotations[i][j], key=self.nerve_annotations[i][j].get)
                    thresh = self.cell_type_confidence[max_vote] if self.cell_type_confidence[max_vote] > 0 else self.confidence_thresh
                    if max_vote != "Others" and self.nerve_annotations[i][j][max_vote] < thresh:
                        self.annotations[i].append("Others")
                        self.confidence[i].append([192, 192, 192])
                    else:
                        self.annotations[i].append(max_vote)
                        self.confidence[i].append(number_to_rgb(self.nerve_annotations[i][j][max_vote]))
        
        else:
            raise ValueError("No predictions to merge")
        
        if self.extra_cell_types:
            self._find_extra_cell_types(min_samples=self.min_cells)
        

    def _find_extra_cell_types(self, root_cell_type="Others", min_samples=10):
        # cluster others
        intensity_others = []
        indices = []
        for i in range(len(self.annotations)):
            for j in range(len(self.annotations[i])):
                # get its intensity
                if self.annotations[i][j] == root_cell_type:
                    intensity_others.append(self.preprocessor.intensity_full[i][j])
                    indices.append([i, j])

        if len(intensity_others) > 0:
            intensity_others = np.array(intensity_others)
            reducer = umap.UMAP(n_components=5)
            embedding = reducer.fit_transform(intensity_others)
            # kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding)
            clustering_model = HDBSCAN(min_cluster_size=min_samples).fit(embedding)
            for i in range(len(clustering_model.labels_)):
                if clustering_model.labels_[i] != -1:
                    self.annotations[indices[i][0]][indices[i][1]] = f"Additional type {clustering_model.labels_[i]}"
                    self.confidence[indices[i][0]][indices[i][1]] = [192, 192, 192]
                else:
                    self.annotations[indices[i][0]][indices[i][1]] = "Others"
                    self.confidence[indices[i][0]][indices[i][1]] = [192, 192, 192]
            

    def _get_unique_cell_types(self):
        return np.unique(self.annotations)
    
    def get_cell_type_names(self):
        txt = ""
        for i in range(len(self.cell_types)):
            txt += f"{i}: {self.cell_types[i]}"
            if i % 5 == 4:
                txt += "\n"
            else:
                txt += "  "
        return txt


    def generate_heatmap(self, integrate=False):
        if len(self.annotations) == 0:
            raise ValueError("No annotations to generate heatmap")
        if integrate:
            temp = []
            for i in range(len(self.annotations)):
                temp += self.annotations[i]
            celltypes = np.unique(temp)
            colormap = np.zeros((len(celltypes), len(self.preprocessor.intensity_full[0][0])))
            for j in range(len(celltypes)):
                temp = []
                for i in range(len(self.annotations)):
                    indices = [k for k in range(len(self.annotations[i])) if self.annotations[i][k] == celltypes[j]]
                    for k in indices:
                        temp.append(self.preprocessor.intensity_full[i][k])
                colormap[j] = np.mean(temp, axis=0)
            # save the heatmap
            f = os.path.join(self.result_dir, f"{self.batch_id}_Integrated_heatmap.png")
            plt.figure(figsize=(colormap.shape[1] // 4, colormap.shape[0] // 4))
            sns.heatmap(colormap, cmap='vlag', xticklabels=self.channel_parser.markers, yticklabels=celltypes, linewidth=.5)
            plt.tight_layout()
            plt.savefig(f)
            plt.close()
        else:
            for i in range(len(self.annotations)):
                celltypes = np.unique(self.annotations[i])
                colormap = np.zeros((len(celltypes), len(self.preprocessor.intensity_full[0][0])))
                for j in range(len(celltypes)):
                    # get indices of the cell type
                    indices = [k for k in range(len(self.annotations[i])) if self.annotations[i][k] == celltypes[j]]
                    temp = []
                    assert len(self.preprocessor.intensity_full[i]) == len(self.annotations[i])
                    for k in indices:
                        temp.append(self.preprocessor.intensity_full[i][k])
                    colormap[j] = np.mean(temp, axis=0)
                # save the heatmap
                f = os.path.join(self.result_dir, f"{self.batch_id}_heatmap_{i}.png")
                sns.heatmap(colormap, cmap='vlag', xticklabels=self.channel_parser.markers, yticklabels=celltypes, linewidth=.5)
                plt.tight_layout()
                plt.savefig(f)
                plt.close()
                
                


    def umap_visualization(self):
        if len(self.annotations) == 0:
            raise ValueError("No annotations to visualize")
        intensity_full = self.preprocessor.intensity_full
        intensity_full = np.concatenate(intensity_full, axis=0)

        colors = {f"{self.cell_types[i]}": rgb_to_hex(self.colors[i]) for i in range(len(self.cell_types))}
        annotations = []
        for i in range(len(self.annotations)):
            annotations += self.annotations[i]
        assert len(intensity_full) == len(annotations)

        reducer = umap.UMAP()
        embedding = reducer.fit_transform(intensity_full)
        f = os.path.join(self.result_dir, f"{self.batch_id}_umap.png")
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=annotations, palette=colors, marker=".", s=15)
        # legend off
        plt.legend([],[], frameon=False)
        plt.savefig(f)
        plt.close()
                

    def export_annotations(self):
        if len(self.annotations) == 0:
            raise ValueError("No annotations to export")
        all_annotations = []

        for i in range(len(self.annotations)):
            temp = []
            f = os.path.join(self.result_dir, f"{self.batch_id}_annotation_{i}.txt")
            with open(f, "w") as file:
                for j, key in enumerate(self.preprocessor.cell_pos_dict[i].keys()):
                    file.write(f"Cell {key}: {self.annotations[i][j]}\n")
                    cell_type_int = np.where(self.cell_types == self.annotations[i][j])[0][0]
                    conf = self.confidence[i][j]
                    # get coordinates
                    row, col = self.preprocessor.cell_pos_dict[i][j + 1]

                    dict_ = {"Cell ID": key, "Cell type": cell_type_int, "Confidence": conf, "Row": row, "Column": col}
                    temp.append(dict_)
                all_annotations.append(temp)

        f = os.path.join(self.temp_dir, f"{self.batch_id}_annotation.pkl")
        with open(f, "wb") as file:
            pickle.dump(all_annotations, file)
                
    def tissue_region_analysis(self, n):
        self.n_regions = n
        f = os.path.join(self.temp_dir, f"{self.batch_id}_annotation.pkl")
        self.tissue_regions = _tissue_region_partition(n, f)

    def colorize(self):
        colors = self.colors
        if len(self.preprocessor.masks) == 0:
            raise ValueError("No masks to colorize")
        if len(self.annotations) == 0:
            raise ValueError("No annotations to colorize")
        

        for i in range(len(self.preprocessor.masks)):
            mask = self.preprocessor.masks[i]
            colormap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colormap2 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            colormap3 = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

            tissuemap = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            tissuemap2 = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            tissue_colors = get_colors(self.n_regions)
            for j in range(1, mask.max() + 1):
                celltype_pred = np.where(self.cell_types == self.annotations[i][j - 1])[0][0]
                row, col = self.preprocessor.cell_pos_dict[i][j]
                colormap[row, col, :] = colors[celltype_pred]
                colormap2[row, col, :] = self.confidence[i][j - 1]
                colormap3[row, col] = celltype_pred + 1

                tissuemap[row, col, :] = tissue_colors[self.tissue_regions[i][j]]
                tissuemap2[row, col] = self.tissue_regions[i][j] + 1
            
            # save the colorized mask
            f = os.path.join(self.result_dir, f"{self.batch_id}_colorized_annotation_{i}.png")
            Image.fromarray(colormap).save(f)

            f = "./src/multiplexed_image_annotator/cell_type_annotation/_working_dir_temp/output_img.png"
            Image.fromarray(colormap3).save(f)

            f = os.path.join(self.result_dir, f"{self.batch_id}_confidence_{i}.png")
            Image.fromarray(colormap2).save(f)

            f = os.path.join(self.result_dir, f"{self.batch_id}_tissue_region_{i}.png")
            Image.fromarray(tissuemap).save(f)

            f = "./src/multiplexed_image_annotator/cell_type_annotation/_working_dir_temp/output_img_2.png"
            Image.fromarray(tissuemap2).save(f)


    def cell_type_composition(self, reduction=True, integrate=False):
        if len(self.annotations) == 0:
            raise ValueError("No annotations to analyze")
        
        if integrate:
            N = 0
            composition = {k: 0 for k in self.cell_types}
            for i in range(len(self.annotations)):
                for j in range(len(self.annotations[i])):
                    composition[self.annotations[i][j]] += 1
                    N += 1
            if reduction:
                for k in composition:
                    composition[k] /= N

            fig = plt.figure()
            ax = fig.add_subplot(111)
            colors = [rgb_to_hex(self.colors[i]) for i in range(len(self.colors))]
            ax.pie(composition.values(), colors=colors)
            legend = [f"{k} ({composition[k] * 100:.2f} %)" for k in composition.keys()]
            # put legend outside the plot
            plt.legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))
            ax.axis('equal')
            plt.tight_layout()
            f = os.path.join(self.result_dir, f"{self.batch_id}_integrated_cell-type_composition.png")
            plt.savefig(f)
            plt.close()

        else:
            for i in range(len(self.annotations)):
                N = 0
                temp = {k: 0 for k in self.cell_types}
                for j in range(len(self.annotations[i])):
                    temp[self.annotations[i][j]] += 1
                    N += 1
                if reduction:
                    for k in temp:
                        temp[k] /= N


                fig = plt.figure()
                ax = fig.add_subplot(111)
                colors = [rgb_to_hex(self.colors[i]) for i in range(len(self.colors))]
                ax.pie(temp.values(), colors=colors)
                legend = [f"{k} ({temp[k] * 100:.2f} %)" for k in temp.keys()]
                # put legend outside the plot
                plt.legend(legend, loc="center left", bbox_to_anchor=(1, 0.5))
                ax.axis('equal')
                plt.tight_layout()
                f = os.path.join(self.result_dir, f"{self.batch_id}_cell-type_composition_{i}.png")
                plt.savefig(f)
                plt.close()

            
    def clear_tmp(self):
        for f in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, f))

