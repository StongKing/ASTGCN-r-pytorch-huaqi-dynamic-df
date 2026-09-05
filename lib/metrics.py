# -*- coding:utf-8 -*-

import numpy as np
import torch


# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         if np.isnan(null_val):
#             mask = ~np.isnan(y_true)
#         else:
#             mask = np.not_equal(y_true, null_val)
#         mask = mask.astype('float32')
#         mask /= np.mean(mask)
#         mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
#                                 y_true))
#         mape = np.nan_to_num(mask * mape)
#         return np.mean(mape)



# def masked_mape_np(y_true, y_pred, null_val=np.nan):
#     """
#     NumPy 版 EasyTorch 风格 masked MAPE
#     1. 过滤 null_val 及极小真值
#     2. 分母截断 min_denom 防止爆炸
#     3. 只在有效位上求平均
#     """
#     min_denom = 1e-4
#     y_true = np.asarray(y_true, dtype=np.float32)
#     y_pred = np.asarray(y_pred, dtype=np.float32)
#     # 1. 有效掩码
#     if np.isnan(null_val):
#         mask = ~np.isnan(y_true)
#     else:
#         mask = y_true != null_val
#     mask &= np.abs(y_true) >= min_denom        # 额外截断极小值
#     if not mask.any():
#         return 0.0
#     # 2. 计算 MAPE（仅有效位）
#     mape = np.abs((y_pred - y_true) / np.clip(y_true, a_min=min_denom, a_max=None))
#     mape = mape * mask                         # 无效位置 0
#     print("valid samples:", mask.sum(), "min |y|:", np.abs(y_true[mask]).min())
#     # 3. 只在有效样本上求平均
#     return float(mape.sum() / mask.sum())




def masked_mape_np(y_true, y_pred, null_val: float = 0.0):
    """
    Masked MAPE like EasyTorch / BasicTS.
    y_true, y_pred: numpy arrays, same shape.
    null_val: value in y_true to be ignored (e.g. 0.0). If np.nan, use isnan mask.
    Returns scalar MAPE (not percentage).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # 构造掩码
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = (y_true != null_val)

        # 如果掩码全为 False（没有有效值），直接返回 np.nan
        if not np.any(mask):
            return np.nan

        mask = mask.astype('float32')
        # 归一化掩码（与 EasyTorch 常用实现一致）
        mask_mean = np.mean(mask)
        if mask_mean == 0:
            return np.nan
        mask /= mask_mean

        # 计算 mape 分子/分母
        mape = np.abs((y_pred - y_true).astype('float32') / (y_true.astype('float32')))
        # 应用掩码并把 NaN/inf 转为 0
        mape = np.nan_to_num(mask * mape, nan=0.0, posinf=0.0, neginf=0.0)

        return np.mean(mape)


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # print(mask.sum())
    # print(mask.shape[0]*mask.shape[1]*mask.shape[2])
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# 修改部分
def mse_loss(preds, labels):
    # 计算预测值和真实标签之间的平方差
    loss = (preds - labels) ** 2 + sum(preds)

    # 返回损失的平均值
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mae_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'),)
        # np.nan_to_num()用零替换NaN，用最大的有限数替换无穷大
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)


def masked_rmse_test(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            # null_val=null_val
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = ((y_pred - y_true) ** 2)
        mse = np.nan_to_num(mask * mse)
        return np.sqrt(np.mean(mse))