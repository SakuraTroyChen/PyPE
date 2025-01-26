import scipy.ndimage

import cv2
import numpy as np

from skimage import transform as skimage_transform
import seaborn as sns
import scipy
from matplotlib import pyplot as plt
import torch.nn.functional as F
import ipdb
import json
from tqdm import tqdm
import torch
import os
import math
from llava.cca_utils.gilbert import gilbert2d
import matplotlib.colors as mcolors
from llava.mm_utils import (
    tokenizer_image_token,
)
from llava.constants import IMAGE_TOKEN_INDEX

def visualize_full_attention(
    model,
    tokenizer,
    input_ids,
    qu,
    image_path,
    image,
    image_tensor,
    index,
    layer_wise=False,
    save_path="attention_visualization/test",
):

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
            )
    qu_append = tokenizer.batch_decode(out[0], skip_special_tokens=True)[0].strip()
    print(qu_append)
    print("\n")
    qu = qu + qu_append
    print(qu)
    print("\n")

    input_ids_qu = (
        tokenizer_image_token(qu, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                input_ids_qu,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=1024,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

    attns = [attn.clone() for attn in out[0]["attentions"][0]]
    # [layer_id, bs, head_id, qk, v]

    # for i in range(len(attns)):
    #     for j in range(len(attns[i])):
    #         attns[i][j] = attns[i][j] * 5
    #         attns[i][j] = attns[i][j].clamp(0, 1)

    p_before, p_after = qu.split("<image>")
    p_before_tokens = (
        tokenizer(p_before, return_tensors="pt", add_special_tokens=False)
        .to("cuda")
        .input_ids
    )
    p_after_tokens = (
        tokenizer(p_after, return_tensors="pt", add_special_tokens=False)
        .to("cuda")
        .input_ids
    )
    p_before_tokens = tokenizer.convert_ids_to_tokens(p_before_tokens[0].tolist())
    p_after_tokens = tokenizer.convert_ids_to_tokens(p_after_tokens[0].tolist())

    bos = torch.ones([1, 1], dtype=torch.int64, device="cuda") * tokenizer.bos_token_id
    bos_tokens = tokenizer.convert_ids_to_tokens(bos[0].tolist())

    print(qu)
    print(p_before_tokens)
    print(p_after_tokens)
    print(bos_tokens)

    NUM_IMAGE_TOKENS = 576
    rows_per_group = 8
    num_groups = int(NUM_IMAGE_TOKENS // rows_per_group)

    tokens = bos_tokens + p_before_tokens + ["img_token"] * num_groups + p_after_tokens
    seq_len = len(tokens)
    len1 = len(bos_tokens + p_before_tokens)
    len2 = len(
        bos_tokens
        + p_before_tokens
        + ["img_token"] * num_groups
        + p_after_tokens[: p_after_tokens.index(":") + 1]
    )
    # print(len(tokens))
    print(len1, len2)

    tokens = [str(idx) + "-" + token for idx, token in enumerate(tokens)]
    print(tokens)

    for layer in range(len(attns)):
        attn_layer = attns[layer].max(1).values.data.squeeze()

        attn_layer_row = attn_layer[len1 : len1 + NUM_IMAGE_TOKENS, :]

        new_attn_layer_row = torch.zeros((num_groups, attn_layer_row.shape[1]))
        # 对每组进行求和
        for i in range(num_groups):
            start_row = i * rows_per_group
            end_row = start_row + rows_per_group
            new_attn_layer_row[i, :] = attn_layer_row[start_row:end_row, :].sum(0)

        # 拼回原矩阵
        aggregated_row_attn_layer = torch.cat(
            [
                attn_layer[:len1, :],
                new_attn_layer_row.to(attn_layer.device),
                attn_layer[len1 + NUM_IMAGE_TOKENS :, :],
            ]
        )

        attn_layer_col = aggregated_row_attn_layer[:, len1 : len1 + NUM_IMAGE_TOKENS]

        new_attn_layer_col = torch.zeros((attn_layer_col.shape[0], num_groups))
        for i in range(num_groups):
            start_col = i * rows_per_group
            end_col = start_col + rows_per_group
            new_attn_layer_col[:, i] = attn_layer_col[:, start_col:end_col].sum(1)

        aggreagated_attn_layer = torch.cat(
            [
                aggregated_row_attn_layer[:, :len1],
                new_attn_layer_col.to(attn_layer.device),
                aggregated_row_attn_layer[:, len1 + NUM_IMAGE_TOKENS :],
            ],
            dim=1,
        )

        aggreagated_attn_layer = aggreagated_attn_layer / aggreagated_attn_layer.sum(
            -1, keepdim=True
        )

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号
        plt.rcParams["xtick.direction"] = "in"

        # plt.rcParams['ytick.direction'] = 'in'
        def draw(data, x, y, ax):
            sns.heatmap(
                data,
                xticklabels=x,
                square=True,
                yticklabels=y,
                vmin=0,
                vmax=1.0,
                cbar=False,
                ax=ax,
            )

        fig, axs = plt.subplots(1, 1, figsize=(100, 100))  # 布置画板
        draw(5 * aggreagated_attn_layer.cpu().numpy(), x=tokens, y=tokens, ax=axs)
        plt.show()
        plt.savefig(
            f"{save_path}/test_{index}_layer{layer}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


def interpolate_attention_scores_to_image(score, target_shape):
    # 添加批量和通道维度以符合 interpolate 的输入要求
    score = score.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 24, 24)

    # 使用最近邻插值将score插值到目标尺寸
    interpolated_score = F.interpolate(score, size=target_shape, mode="nearest")

    # 移除批量和通道维度
    interpolated_score = interpolated_score.squeeze(0).squeeze(0)

    return interpolated_score


def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode="constant")
    if blur:
        attMap = scipy.ndimage.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap("jet")
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = (
            1 * (1 - attMap**0.7).reshape(attMap.shape + (1,)) * img
            + (attMap**0.7).reshape(attMap.shape + (1,)) * attMapV
        )
    return attMap


def visualize_attention(
    model,
    tokenizer,
    input_ids,
    image_path,
    image,
    image_tensor,
    idx,
    layer_wise=False,
    save_path="attention_visualization/",
):
    # 创建保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_scores, attn_layers_weights = get_attention_score(
        model,
        tokenizer,
        input_ids,
        image,
        image_tensor,
        layer_wise=layer_wise,
    )

    if layer_wise:
        plt.bar(range(32), attn_layers_weights)
        plt.xlabel("Layer Index")
        plt.ylabel("Attention Weight")
        plt.title("Attention Weight per Layers")
        plt.tight_layout()
        plt.savefig(
            f"{save_path}/{idx}_attention_weight.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))

    for layer_idx, layer in enumerate(output_scores):
        interpolation = interpolate_attention_scores_to_image(layer, (336, 336))

        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255
        ax.imshow(rgb_image, aspect="auto")

        gradcam_image = getAttMap(
            rgb_image, interpolation.cpu().numpy().astype(np.float16)
        )

        ax.imshow(gradcam_image)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        if layer_wise:
            plt.savefig(
                f"{save_path}/{idx}_layer_{layer_idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
        else:
            plt.savefig(
                f"{save_path}/{idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

    plt.close()

    # for layer_idx, layer in enumerate(output_scores):
    #     layer = layer.detach().cpu().numpy()
    #     txt_img_if_max = layer.max()
    #     txt_img_if_min = layer.min()
    #     norm_txt_img_if = (layer - txt_img_if_min) / (txt_img_if_max - txt_img_if_min)
    #     vmin = norm_txt_img_if.min()
    #     vmax = norm_txt_img_if.max()
    #     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #     plt.imshow(norm_txt_img_if, cmap="viridis", interpolation="nearest", norm=norm)
    #     plt.colorbar()
    #     plt.axis("off")
    #     if layer_wise:
    #         plt.savefig(
    #             f"{save_path}/{idx}_layer_{layer_idx}_for_rope.png",
    #             bbox_inches="tight",
    #             pad_inches=0,
    #         )
    #     else:
    #         plt.savefig(
    #             f"{save_path}/{idx}_for_rope.png",
    #             bbox_inches="tight",
    #             pad_inches=0,
    #         )

    #     plt.close()


def get_attention_score(
    model,
    tokenizer,
    input_ids,
    image,
    image_tensor,
    layer_wise=False,
):

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            do_sample=False,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

    # response_start = torch.where(input_ids == 13566)[1][0]
    img_s = torch.where(input_ids == -200)[1][0]

    attn_scores = []

    if not layer_wise:
        # aggregation of attentions
        attn_layers = [sum(outputs[0]["attentions"][0])]
    else:
        # layer-wise attention
        attn_layers = [outputs[0]["attentions"][0][layer] for layer in range(32)]

    response_sum_attn = sum(outputs[0]["attentions"][0])[0, :, img_s + 576 :]
    response_sum_attn_score = response_sum_attn[:, :, img_s : img_s + 576].mean(
        dim=[0, 1]
    )
    attn_layers_weights = []
    for layer in attn_layers:
        score = layer[0, :, img_s + 576 :, img_s : img_s + 576]

        # 平均了32个头的attention
        score = score.mean(dim=[0, 1]).reshape(24, 24)

        # 输出求和得到的attention中每层attention的权重占比
        attn_layers_weight = sum(score.flatten()) / sum(
            response_sum_attn_score.flatten()
        )
        attn_layers_weights.append(attn_layers_weight.cpu().numpy())

        attn_scores.append(score)

    return attn_scores, attn_layers_weights
