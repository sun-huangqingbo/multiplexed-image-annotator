from functools import partial

import torch
import torch.nn as nn
import numpy as np
import os
from timm.models.vision_transformer import PatchEmbed, Block



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
    else:
        grid_h = np.arange(grid_size[0], dtype=np.float32)
        grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    if isinstance(grid_size, int):
        grid = grid.reshape([2, 1, grid_size, grid_size])
    else:
        grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.in_chans = in_chans
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, in_channel, H, W)
        x: (N, L, patch_size**2 *in_channel)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        # h = w = imgs.shape[2] // p
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans ))
        return x

    def unpatchify(self, x, shape):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1]**.5)
        # h = 2
        # w = 5
        h, w = shape
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, len_keep, noise):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  
        
        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(x.device) 
        # SQ notes: ids_restore is: for each entry, its order in the original array
        # e.g. [3,1,2,4] -- > (ascending)ids_restore [2,0,1,3] vs ids_shuffle[1,2,0,3]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep].to(x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # SQ notes: x_masked use the shuffled

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # SQ notes: this line of code use ids_restore to get the unshuffled binary mask
        # e.g. mask = [[0,0,0,1]], ids_restore = [2,3,1,0] --> mask after gather = [0,1,0,0]

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, len_keep, noise):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, len_keep, noise)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, len_keep, noise):
        latent, mask, ids_restore = self.forward_encoder(x, len_keep, noise)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, mask
    

class MarkerImputer():
    def __init__(self, channel_index, device, panel=""):
        if panel == "immune_full" and os.path.exists(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_full_impute.pth"):
            checkpoint = torch.load(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_full_impute.pth", map_location=device)["model"]
            img_size = (120, 200)
            channel_number = 15
            self.shape = (3, 5)
        elif panel == "immune_extended" and os.path.exists(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_extended_impute.pth"):
            checkpoint = torch.load(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_extended_impute.pth", map_location=device)["model"]
            img_size = (80, 200)
            channel_number = 10
            self.shape = (2, 5)
        elif panel == "immune_base" and os.path.exists(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_base_impute.pth"):
            checkpoint = torch.load(r"src/multiplexed_image_annotator/cell_type_annotation/models/immune_base_impute.pth", map_location=device)["model"]
            img_size = (40, 280)
            channel_number = 7
            self.shape = (1, 7)
        else:
            raise ValueError("Invalid panel")
        
        self.device = device
        
        self.model = MaskedAutoencoderViT(
            img_size=img_size, in_chans=1,
            patch_size=40, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=8,
            mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()

        

        self.channel_index = channel_index
        self.channel_number = channel_number

    def impute(self, data, batch_size=1):
        h, w = self.shape # 3, 5
        with torch.no_grad():
            data_2 = torch.zeros(data.shape[0], 1, h * data.shape[2], w * data.shape[3])
            for i in range(h):
                for j in range(w):
                    data_2[:, 0, i * data.shape[2]: (i + 1) * data.shape[2], j * data.shape[3]: (j + 1) * data.shape[3]] = data[:, i * w + j, :, :]
            data_2 = data_2.to(self.device)

            noise = torch.ones(data_2.shape[0], h * w) * 0.8
            for i in range(self.channel_number):
                if i in self.channel_index:
                    noise[:, i] = 0.1

            for i in range(len(data_2) // batch_size + 1):
                x = data_2[i * batch_size: (i + 1) * batch_size].to(self.device)
                noise_ = noise[i * batch_size: (i + 1) * batch_size].to(self.device)
                pred, mask = self.model(x, len(self.channel_index), noise_)
                pred = self.model.unpatchify(pred, self.shape)
                mask = mask.detach()
                mask = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_embed.patch_size[0]**2 *self.model.in_chans)
                mask = self.model.unpatchify(mask, self.shape)
                out_img = x * (1 - mask) + pred * mask
                out_img = out_img.cpu()
                if i == 0:
                    out_imgs = out_img
                else:
                    out_imgs = torch.cat((out_imgs, out_img), dim=0)
        del data_2

        for i in range(h):
            for j in range(w):
                data[:, i * w + j, :, :] = out_imgs[:, 0, i * data.shape[2]: (i + 1) * data.shape[2], j * data.shape[3]: (j + 1) * data.shape[3]]


        return data

