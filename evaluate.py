#!/usr/bin/env python3
"""
评估脚本：量化一个已训练 ContactNet checkpoint 的表现（顶点级 contact 二分类）。

你的需求（已对齐）：
- GT 格式与训练一致：每个样本目录内有 `contact.json`，其中人体 GT 为前 10475 个顶点（与 `SmplContactDataset` 读取逻辑一致）。
- 只需最简 CLI：--config --checkpoint --data_root --out_dir
- 使用 data_root 下“全部样本”评估（不区分 train/val/test，不做数据增强）
- threshold 策略：多阈值扫描（输出 sweep 曲线 + 推荐阈值）
- 指标：最小集 + 增强集；分部位指标暂不实现，但会在代码里留 TODO

输出（写入 out_dir）：
- metrics_summary.json
  - 全局 micro/macro 指标、增强指标（如 Brier/PR-AUC/ROC-AUC 的近似值）、推荐阈值等
- threshold_sweep.json
  - 多阈值扫描曲线（precision/recall/f1/iou/...），以及若干 F-beta 下的最佳阈值
- metrics_per_sample.csv
  - 在“默认推荐阈值”下逐样本指标（便于排序找失败 case）

实现说明（重要）：
1) 为了避免在大数据集上做 O(T*N) 的直接阈值扫描，本脚本用“高分辨率概率直方图”近似统计：
   - 设 bins=10001（分辨率 1e-4），单次遍历所有顶点概率，累积正/负样本在各概率 bin 的计数
   - 再通过从高到低的前缀和，快速得到任意阈值下的 TP/FP/FN/TN（近似误差 <= 1 bin）
2) 由于你更看重召回，但希望可调节 precision/recall 的权衡：
   - 我们报告并推荐 “F-beta 最优阈值”，beta 越大越偏向 recall
   - beta 默认值从 config 读取（config.evaluation.fbeta_beta），缺省为 2.0
   - 也会同时给出若干常用 beta（0.5/1/2/4）的推荐阈值，便于你选择

TODO（后续可做）：
- 分部位/手部等区域指标：需要稳定的 vertex index 分区映射（或 face/vertex region 文件）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.dataset import SmplContactDataset, collate_fn
from models.contact_net import ContactNet


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(path: str) -> str:
    if path is None:
        return path
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_plots(
    *,
    out_dir: str,
    thresholds: np.ndarray,
    metrics_curve: List[Dict[str, float]],
    best_by_beta: List[Dict[str, Any]],
    best_default: Dict[str, Any],
    beta_default: float,
    by_category_micro: Optional[Dict[str, Any]] = None,
    by_category_samples: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    将重要评估结果可视化为 PNG 并保存到 out_dir。

    产物（尽量少且信息密度高）：
    - plot_threshold_metrics.png: threshold vs precision/recall/f1/iou（标注推荐阈值）
    - plot_pr_curve.png: PR 曲线（recall-precision），标注推荐阈值点
    - plot_fbeta_sweep.png: 多 beta 的 F-beta vs threshold，并标注各自最优点
    - plot_by_category.png: (可选) 按类别的 micro recall/precision 概览（Top-N）
    """
    saved: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless save
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plot] Warning: matplotlib not available, skip plots. error={e}")
        return saved

    # Extract arrays
    th = thresholds.astype(np.float32)
    prec = np.asarray([m["precision"] for m in metrics_curve], dtype=np.float32)
    rec = np.asarray([m["recall"] for m in metrics_curve], dtype=np.float32)
    f1 = np.asarray([m["f1"] for m in metrics_curve], dtype=np.float32)
    iou = np.asarray([m["iou"] for m in metrics_curve], dtype=np.float32)

    chosen_t = float(best_default.get("threshold", 0.5))
    chosen_i = int(best_default.get("index", int(np.argmin(np.abs(th - chosen_t)))))

    # 1) threshold metrics
    try:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.plot(th, rec, label="Recall", linewidth=2)
        ax.plot(th, prec, label="Precision", linewidth=2)
        ax.plot(th, f1, label="F1", linewidth=2)
        ax.plot(th, iou, label="IoU", linewidth=2)
        ax.axvline(chosen_t, color="k", linestyle="--", linewidth=1.5, label=f"Recommended t={chosen_t:.3f} (Fβ, β={beta_default:g})")
        ax.scatter([th[chosen_i]], [rec[chosen_i]], s=60, color="tab:blue")
        ax.scatter([th[chosen_i]], [prec[chosen_i]], s=60, color="tab:orange")
        ax.set_title("Threshold sweep (vertex-level)")
        ax.set_xlabel("threshold (pred = prob > threshold)")
        ax.set_ylabel("metric")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, "plot_threshold_metrics.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)
    except Exception as e:
        print(f"[Plot] threshold metrics plot failed: {e}")

    # 2) PR curve
    try:
        # Using sweep points: x=recall, y=precision
        fig = plt.figure(figsize=(6.5, 6.0))
        ax = fig.add_subplot(111)
        ax.plot(rec, prec, linewidth=2)
        ax.scatter([rec[chosen_i]], [prec[chosen_i]], s=70, color="red", label=f"Recommended t={chosen_t:.3f}")
        ax.set_title("PR curve (approx from sweep)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, "plot_pr_curve.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)
    except Exception as e:
        print(f"[Plot] PR plot failed: {e}")

    # 3) F-beta sweep (multi-beta)
    try:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        betas = []
        for binfo in best_by_beta:
            try:
                betas.append(float(binfo.get("beta")))
            except Exception:
                continue
        # de-dup & keep stable
        betas = list(dict.fromkeys(betas))
        for b in betas[:6]:  # avoid overcrowding
            fbeta_curve = np.asarray([_f_beta(float(p), float(r), float(b)) for p, r in zip(prec.tolist(), rec.tolist())], dtype=np.float32)
            ax.plot(th, fbeta_curve, linewidth=2, label=f"Fβ (β={b:g})")
            # mark best
            best = None
            for bi in best_by_beta:
                if float(bi.get("beta", -1)) == float(b):
                    best = bi
                    break
            if best is not None:
                bt = float(best.get("threshold", 0.5))
                bj = int(best.get("index", int(np.argmin(np.abs(th - bt)))))
                ax.scatter([th[bj]], [fbeta_curve[bj]], s=60)
        ax.axvline(chosen_t, color="k", linestyle="--", linewidth=1.2)
        ax.set_title("F-beta sweep (recall-preference control)")
        ax.set_xlabel("threshold")
        ax.set_ylabel("Fβ")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=9, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_dir, "plot_fbeta_sweep.png")
        fig.savefig(p, dpi=160)
        plt.close(fig)
        saved.append(p)
    except Exception as e:
        print(f"[Plot] F-beta sweep plot failed: {e}")

    # 4) By-category overview (optional)
    try:
        if by_category_micro:
            rows = []
            for cat, m in by_category_micro.items():
                if not isinstance(m, dict):
                    continue
                r = float(m.get("recall", 0.0))
                p = float(m.get("precision", 0.0))
                gt_ratio = float(m.get("gt_pos_ratio", 0.0))
                pred_ratio = float(m.get("pred_pos_ratio", 0.0))
                n = int(by_category_samples.get(cat, 0)) if isinstance(by_category_samples, dict) else 0
                rows.append((cat, n, r, p, gt_ratio, pred_ratio))
            # filter unknown / small
            rows = [x for x in rows if x[0]]
            if rows:
                # sort:
                # - if we have sample counts, sort by count desc, then recall asc (surface worst within big cats)
                # - otherwise (all n==0), sort by recall asc directly
                if sum(x[1] for x in rows) > 0:
                    rows.sort(key=lambda x: (-x[1], x[2]))
                else:
                    rows.sort(key=lambda x: x[2])
                topk = rows[:20]
                cats = [x[0] for x in topk][::-1]
                ns = [x[1] for x in topk][::-1]
                rs = [x[2] for x in topk][::-1]
                ps = [x[3] for x in topk][::-1]
                gts = [x[4] for x in topk][::-1]
                preds = [x[5] for x in topk][::-1]

                fig = plt.figure(figsize=(10, 0.4 * len(cats) + 2.5))
                ax = fig.add_subplot(111)
                y = np.arange(len(cats))
                ax.barh(y - 0.18, rs, height=0.35, label="Recall")
                ax.barh(y + 0.18, ps, height=0.35, label="Precision")
                ax.set_yticks(y)
                # 为避免“柱子为 0 看起来像缺失”，在标签里显示 gt/pred 比例，并在右侧标注数值。
                ax.set_yticklabels(
                    [
                        f"{c} (n={n}, gt={gt:.4f}, pred={pr:.4f})"
                        for c, n, gt, pr in zip(cats, ns, gts, preds)
                    ],
                    fontsize=9,
                )
                ax.set_xlim(0.0, 1.0)
                ax.set_xlabel("metric (micro, at recommended threshold)")
                ax.set_title("By-category overview (Top 20 by sample count)")
                ax.grid(True, axis="x", alpha=0.3)
                ax.legend(loc="lower right", fontsize=9)

                # 数值标注：即使为 0，也会显示 “R=0.00 P=0.00”，避免误解为缺失
                for yi, r, p in zip(y.tolist(), rs, ps):
                    x_text = float(max(r, p)) + 0.01
                    x_text = min(x_text, 0.98)
                    ax.text(
                        x_text,
                        yi,
                        f"R={float(r):.2f} P={float(p):.2f}",
                        va="center",
                        ha="left",
                        fontsize=8,
                        color="black",
                    )

                fig.tight_layout()
                pth = os.path.join(out_dir, "plot_by_category.png")
                fig.savefig(pth, dpi=160)
                plt.close(fig)
                saved.append(pth)
    except Exception as e:
        print(f"[Plot] by-category plot failed: {e}")

    return saved


def _load_checkpoint_state_dict(checkpoint_path: str, device: torch.device) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    兼容两种 checkpoint：
    - train.py 保存的 dict：包含 model_state_dict / epoch / best_val_loss ...
    - 纯 state_dict：{param_name: tensor, ...}
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        return ckpt, None
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def _to_numpy_probs_and_labels(
    probs_t: torch.Tensor,
    labels_t: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    probs_t: (B, N) float in [0,1]
    labels_t: (B, N) float/bool in {0,1}
    """
    probs = probs_t.detach().float().cpu().numpy().astype(np.float32, copy=False)
    labels = labels_t.detach().cpu().numpy()
    if labels.dtype != np.bool_:
        labels = labels.astype(np.float32, copy=False)
        labels = labels > 0.5
    return probs, labels


def _bincount_idx(idx: np.ndarray, bins: int) -> np.ndarray:
    if idx.size == 0:
        return np.zeros((bins,), dtype=np.int64)
    return np.bincount(idx, minlength=bins).astype(np.int64, copy=False)


def _compute_confusion_from_hist(
    pos_counts: np.ndarray,
    neg_counts: np.ndarray,
    *,
    thresholds: np.ndarray,
) -> Dict[str, Any]:
    """
    用直方图累计的 pos/neg counts，快速计算多阈值下的 TP/FP/FN/TN（近似）。
    thresholds: (T,) in [0,1]
    """
    assert pos_counts.ndim == 1 and neg_counts.ndim == 1
    assert pos_counts.shape == neg_counts.shape
    bins = int(pos_counts.shape[0])

    total_pos = int(pos_counts.sum())
    total_neg = int(neg_counts.sum())
    total = total_pos + total_neg

    # cum_*[k] = sum_{i>=k} counts[i]
    cum_pos = np.cumsum(pos_counts[::-1])[::-1]
    cum_neg = np.cumsum(neg_counts[::-1])[::-1]

    tps: List[int] = []
    fps: List[int] = []
    fns: List[int] = []
    tns: List[int] = []

    for t in thresholds.tolist():
        # pred = (p > t). With histogram bins of width ~ 1/(bins-1),
        # we approximate by selecting bins whose (bin_value) > t.
        if t >= 1.0:
            tp = 0
            fp = 0
        elif t < 0.0:
            tp = total_pos
            fp = total_neg
        else:
            # idx in [0, bins-1] corresponds to prob approx idx/(bins-1)
            # strict ">" => start from floor(t*(bins-1))+1
            idx_thr = int(np.floor(float(t) * float(bins - 1))) + 1
            idx_thr = int(np.clip(idx_thr, 0, bins))  # bins means empty selection
            if idx_thr >= bins:
                tp = 0
                fp = 0
            else:
                tp = int(cum_pos[idx_thr])
                fp = int(cum_neg[idx_thr])

        fn = int(total_pos - tp)
        tn = int(total_neg - fp)
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)

    return {
        "bins": bins,
        "total": total,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "tp": tps,
        "fp": fps,
        "fn": fns,
        "tn": tns,
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0.0 else 0.0


def _metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    pred_pos = tp + fp
    gt_pos = tp + fn
    pred_pos_ratio = _safe_div(pred_pos, tp + tn + fp + fn)
    gt_pos_ratio = _safe_div(gt_pos, tp + tn + fp + fn)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(acc),
        "pred_pos_ratio": float(pred_pos_ratio),
        "gt_pos_ratio": float(gt_pos_ratio),
    }


def _metrics_from_confusion_f(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    """与 `_metrics_from_confusion` 等价，但输入为 float（用于面积加权指标）。"""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    tn = float(tn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    pred_pos = tp + fp
    gt_pos = tp + fn
    denom = tp + tn + fp + fn
    pred_pos_ratio = _safe_div(pred_pos, denom)
    gt_pos_ratio = _safe_div(gt_pos, denom)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(acc),
        "pred_pos_ratio": float(pred_pos_ratio),
        "gt_pos_ratio": float(gt_pos_ratio),
    }


def _f_beta(precision: float, recall: float, beta: float) -> float:
    b2 = float(beta) ** 2
    denom = (b2 * precision + recall)
    if denom <= 0:
        return 0.0
    return float((1.0 + b2) * precision * recall / denom)


def _pick_best_threshold(
    thresholds: np.ndarray,
    metrics_curve: List[Dict[str, float]],
    *,
    beta: float,
) -> Dict[str, Any]:
    """
    选择使 F-beta 最大的阈值。若并列，优先 recall 更高，再优先 precision 更高。
    """
    best_i = 0
    best_score = -1.0
    best_recall = -1.0
    best_precision = -1.0
    for i, m in enumerate(metrics_curve):
        p = float(m["precision"])
        r = float(m["recall"])
        score = _f_beta(p, r, beta)
        if (
            (score > best_score)
            or (score == best_score and r > best_recall)
            or (score == best_score and r == best_recall and p > best_precision)
        ):
            best_i = i
            best_score = score
            best_recall = r
            best_precision = p
    return {
        "beta": float(beta),
        "threshold": float(thresholds[best_i]),
        "f_beta": float(best_score),
        "precision": float(best_precision),
        "recall": float(best_recall),
        "index": int(best_i),
    }


def _approx_pr_auc_from_hist(pos_counts: np.ndarray, neg_counts: np.ndarray) -> float:
    """
    近似 PR-AUC (Average Precision)：
    - 使用 bins 分箱得到的 “阈值 -> (precision, recall)” 曲线做分段面积累计。
    - bins 越大，近似越精细（默认 10001）。
    """
    total_pos = float(pos_counts.sum())
    if total_pos <= 0:
        return 0.0
    cum_pos = np.cumsum(pos_counts[::-1])[::-1].astype(np.float64)
    cum_neg = np.cumsum(neg_counts[::-1])[::-1].astype(np.float64)

    tp = cum_pos
    fp = cum_neg
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / total_pos

    # AP: sum over recall increments * precision (step-wise)
    # As threshold decreases, recall increases. Our arrays are for k increasing threshold cut (from high probs),
    # so recall is non-increasing with k; we reverse to make recall increasing.
    precision_r = precision[::-1]
    recall_r = recall[::-1]
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision_r.tolist(), recall_r.tolist()):
        if r > prev_recall:
            ap += (r - prev_recall) * p
            prev_recall = r
    return float(ap)


def _approx_roc_auc_from_hist(pos_counts: np.ndarray, neg_counts: np.ndarray) -> float:
    """
    近似 ROC-AUC：用 bins 曲线做梯形积分。
    """
    total_pos = float(pos_counts.sum())
    total_neg = float(neg_counts.sum())
    if total_pos <= 0 or total_neg <= 0:
        return 0.0
    cum_pos = np.cumsum(pos_counts[::-1])[::-1].astype(np.float64)
    cum_neg = np.cumsum(neg_counts[::-1])[::-1].astype(np.float64)
    tpr = cum_pos / total_pos
    fpr = cum_neg / total_neg

    # reverse so fpr increases
    tpr_r = tpr[::-1]
    fpr_r = fpr[::-1]
    auc = float(np.trapz(y=tpr_r, x=fpr_r))
    return float(np.clip(auc, 0.0, 1.0))


def _vertex_area_weights(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    计算每个顶点的近似面积权重（把每个三角形面积平均分到三个顶点）。
    vertices: (N, 3) CPU float tensor
    faces: (F, 3) CPU int64 tensor
    return: (N,) CPU float tensor
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    area = 0.5 * torch.norm(cross, dim=-1)  # (F,)
    w = torch.zeros((vertices.shape[0],), dtype=torch.float32)
    share = (area / 3.0).to(torch.float32)
    for j in range(3):
        w.scatter_add_(0, faces[:, j], share)
    return w


@dataclass
class _Agg:
    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def add(self, tp: float, fp: float, fn: float, tn: float) -> None:
        self.tp += float(tp)
        self.fp += float(fp)
        self.fn += float(fn)
        self.tn += float(tn)

    def metrics(self) -> Dict[str, float]:
        # 注意：这里用 float 版本，兼容“计数型 micro”和“面积加权 micro”两种累加器
        return _metrics_from_confusion_f(self.tp, self.fp, self.fn, self.tn)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate ContactNet checkpoint (threshold sweep).")
    parser.add_argument("--config", type=str, required=True, help="YAML config used to build model + preprocessing")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root directory (same structure as training)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for evaluation reports")
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    checkpoint_path = _resolve_project_path(args.checkpoint)
    data_root = _resolve_project_path(args.data_root)
    out_dir = _resolve_project_path(args.out_dir)
    _safe_mkdir(out_dir)

    # -----------------------------------------
    # Fast path: if evaluation outputs already exist in out_dir and match the inputs,
    # skip inference and ONLY generate plots. (No extra CLI flags.)
    # -----------------------------------------
    try:
        sweep_path = os.path.join(out_dir, "threshold_sweep.json")
        summary_path = os.path.join(out_dir, "metrics_summary.json")
        csv_path_cached = os.path.join(out_dir, "metrics_per_sample.csv")
        if os.path.exists(sweep_path) and os.path.exists(summary_path) and os.path.exists(csv_path_cached):
            with open(sweep_path, "r", encoding="utf-8") as f:
                sweep_cached = json.load(f)
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_cached = json.load(f)

            def _norm(p: str) -> str:
                try:
                    return os.path.normpath(str(p))
                except Exception:
                    return str(p)

            same = True
            same = same and (_norm(sweep_cached.get("config")) == _norm(config_path))
            same = same and (_norm(sweep_cached.get("checkpoint")) == _norm(checkpoint_path))
            same = same and (_norm(sweep_cached.get("data_root")) == _norm(data_root))

            if same:
                thresholds = np.asarray(sweep_cached.get("thresholds", []), dtype=np.float32)
                curve = sweep_cached.get("curve", []) or []
                best_by_beta = sweep_cached.get("best_by_fbeta", []) or []
                best_default = sweep_cached.get("best_default", {}) or {}
                beta_default = float(sweep_cached.get("default_beta", 2.0))

                if thresholds.size > 0 and isinstance(curve, list) and len(curve) == int(thresholds.size):
                    print("=" * 70)
                    print("[Cache] Found existing evaluation outputs matching inputs.")
                    print("[Cache] Skipping inference; generating plots only ...")
                    print("=" * 70)
                    plot_paths = _save_plots(
                        out_dir=out_dir,
                        thresholds=thresholds,
                        metrics_curve=curve,
                        best_by_beta=best_by_beta,
                        best_default=best_default,
                        beta_default=float(beta_default),
                        by_category_micro=(summary_cached.get("by_category_micro") if isinstance(summary_cached, dict) else None),
                        by_category_samples=(summary_cached.get("by_category_samples") if isinstance(summary_cached, dict) else None),
                    )
                    for p in plot_paths:
                        print(f"- plot: {p}")
                    return 0
    except Exception as e:
        # Best-effort cache path only; fall back to full evaluation
        print(f"[Cache] Ignored cache due to error: {e}")

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config YAML (expect dict): {config_path}")

    # Resolve important paths and force eval-safe settings
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("training", {})

    config["data"]["root_dir"] = data_root
    if "smplx_model_path" in config["data"]:
        config["data"]["smplx_model_path"] = _resolve_project_path(str(config["data"]["smplx_model_path"]))

    # Avoid any accidental online download in evaluation: backbone weights will be loaded from checkpoint anyway.
    config["model"]["pretrained"] = False

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ContactNet Evaluation (threshold sweep)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data root: {data_root}")
    print(f"Out dir: {out_dir}")

    # Build dataset (all samples under data_root)
    dataset = SmplContactDataset(
        root_dir=data_root,
        smplx_model_path=config["data"]["smplx_model_path"],
        smplx_model_type=config["data"].get("smplx_model_type", "neutral"),
        img_size=tuple(config["data"].get("img_size", [512, 512])),
        split="test",
        augment=False,
        indices=None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty under: {data_root}")

    # Build mapping: sample_id -> category (for per-category summary)
    id_to_category: Dict[str, str] = {}
    try:
        for s in getattr(dataset, "samples", []) or []:
            sid = f"{s.get('category')}_{s.get('id')}"
            id_to_category[str(sid)] = str(s.get("category"))
    except Exception:
        id_to_category = {}

    # Faces for optional area-weighted metrics (CPU tensor)
    faces_cpu: Optional[torch.Tensor] = None
    try:
        faces_cpu = getattr(dataset, "faces", None)
        if isinstance(faces_cpu, torch.Tensor):
            faces_cpu = faces_cpu.detach().cpu().to(torch.int64)
        else:
            faces_cpu = None
    except Exception:
        faces_cpu = None

    # DataLoader
    batch_size = int(config.get("training", {}).get("batch_size", 1))
    num_workers = int(config.get("data", {}).get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model = ContactNet(config).to(device)
    state_dict, ckpt_meta = _load_checkpoint_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluation hyperparam for precision/recall tradeoff
    beta_default = float(config.get("evaluation", {}).get("fbeta_beta", 2.0))
    betas_report = [0.5, 1.0, 2.0, 4.0]
    if beta_default not in betas_report:
        betas_report = [beta_default] + betas_report

    # Threshold sweep settings (user requires sweep)
    thresholds = np.linspace(0.0, 1.0, num=101, dtype=np.float32)  # 0.00, 0.01, ..., 1.00
    bins = 10001  # probability resolution ~ 1e-4

    # PASS 1: accumulate histograms + Brier score
    pos_counts = np.zeros((bins,), dtype=np.int64)
    neg_counts = np.zeros((bins,), dtype=np.int64)
    brier_sum = 0.0
    total_vertices = 0

    print("\n[Pass1/2] Running inference + building probability histograms ...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval pass1"):
            images = batch["image"].to(device)
            vertices = batch["vertices"].to(device)
            normals = batch["normals"].to(device)
            pose_params = batch["pose_params"].to(device)
            K = batch["K"].to(device)
            object_bbox = batch["object_bbox"].to(device)
            mask_dist_field = batch["mask_dist_field"].to(device)
            labels_t = batch["contact_labels"].to(device)

            logits = model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
            probs_t = torch.sigmoid(logits)  # (B, N)

            probs, labels = _to_numpy_probs_and_labels(probs_t, labels_t)
            p_flat = probs.reshape(-1)
            y_flat = labels.reshape(-1).astype(np.float32, copy=False)

            # Brier score accumulation
            brier_sum += float(np.square(p_flat - y_flat).sum())
            total_vertices += int(p_flat.size)

            # Histogram (approx) for threshold sweep
            idx = (p_flat * float(bins - 1)).astype(np.int32)
            idx = np.clip(idx, 0, bins - 1)
            pos_mask = labels.reshape(-1)
            neg_mask = ~pos_mask
            pos_counts += _bincount_idx(idx[pos_mask], bins)
            neg_counts += _bincount_idx(idx[neg_mask], bins)

    brier = float(brier_sum / max(1, total_vertices))
    pr_auc = _approx_pr_auc_from_hist(pos_counts, neg_counts)
    roc_auc = _approx_roc_auc_from_hist(pos_counts, neg_counts)

    conf = _compute_confusion_from_hist(pos_counts, neg_counts, thresholds=thresholds)
    metrics_curve: List[Dict[str, float]] = []
    for tp, fp, fn, tn in zip(conf["tp"], conf["fp"], conf["fn"], conf["tn"]):
        metrics_curve.append(_metrics_from_confusion(int(tp), int(fp), int(fn), int(tn)))

    best_by_beta: List[Dict[str, Any]] = []
    for b in betas_report:
        best_by_beta.append(_pick_best_threshold(thresholds, metrics_curve, beta=float(b)))
    best_default = _pick_best_threshold(thresholds, metrics_curve, beta=float(beta_default))

    # Write threshold sweep report
    sweep_out = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "data_root": data_root,
        "bins": int(bins),
        "thresholds": thresholds.tolist(),
        "curve": metrics_curve,
        "best_by_fbeta": best_by_beta,
        "default_beta": float(beta_default),
        "best_default": best_default,
        "approx": {
            "brier": float(brier),
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
        },
        "counts": {
            "total_vertices": int(total_vertices),
            "total_pos": int(conf["total_pos"]),
            "total_neg": int(conf["total_neg"]),
        },
        "notes": [
            "threshold sweep uses histogram approximation (bins=10001, resolution ~1e-4).",
            "pred is defined as (prob > threshold).",
        ],
    }
    with open(os.path.join(out_dir, "threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep_out, f, indent=2, ensure_ascii=False)

    chosen_threshold = float(best_default["threshold"])
    print("\n" + "-" * 70)
    print(f"Recommended threshold by F-beta (beta={beta_default}): {chosen_threshold:.4f}")
    print(
        f"At that threshold: P={best_default['precision']:.4f}, R={best_default['recall']:.4f}, Fbeta={best_default['f_beta']:.4f}"
    )
    print(f"Approx Brier={brier:.6f}, PR-AUC={pr_auc:.6f}, ROC-AUC={roc_auc:.6f}")
    print("-" * 70)

    # PASS 2: per-sample metrics at chosen threshold (and category summary)
    print("\n[Pass2/2] Re-running inference for per-sample metrics ...")
    per_sample_rows: List[Dict[str, Any]] = []
    macro_metrics: List[Dict[str, float]] = []

    micro = _Agg()
    micro_w = _Agg()  # area-weighted (if faces available)
    by_cat: Dict[str, _Agg] = {}
    by_cat_samples: Dict[str, int] = {}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval pass2"):
            images = batch["image"].to(device)
            vertices = batch["vertices"].to(device)
            normals = batch["normals"].to(device)
            pose_params = batch["pose_params"].to(device)
            K = batch["K"].to(device)
            object_bbox = batch["object_bbox"].to(device)
            mask_dist_field = batch["mask_dist_field"].to(device)
            labels_t = batch["contact_labels"].to(device)
            sample_ids = batch.get("sample_ids", ["unknown"] * images.shape[0])

            logits = model(images, vertices, normals, pose_params, K, object_bbox, mask_dist_field)
            probs_t = torch.sigmoid(logits)  # (B, N)

            probs, labels = _to_numpy_probs_and_labels(probs_t, labels_t)
            preds = probs > float(chosen_threshold)

            for i in range(probs.shape[0]):
                sid = str(sample_ids[i])
                cat = id_to_category.get(sid, "")

                y = labels[i].astype(bool, copy=False)
                p = preds[i].astype(bool, copy=False)

                tp = int(np.logical_and(p, y).sum())
                fp = int(np.logical_and(p, ~y).sum())
                fn = int(np.logical_and(~p, y).sum())
                tn = int(np.logical_and(~p, ~y).sum())

                m = _metrics_from_confusion(tp, fp, fn, tn)
                macro_metrics.append(m)

                # micro accumulate
                micro.add(tp, fp, fn, tn)
                if cat:
                    by_cat.setdefault(cat, _Agg()).add(tp, fp, fn, tn)
                    by_cat_samples[cat] = int(by_cat_samples.get(cat, 0)) + 1

                # area-weighted metrics (optional)
                w_metrics: Optional[Dict[str, float]] = None
                if faces_cpu is not None:
                    try:
                        v_cpu = vertices[i].detach().cpu().to(torch.float32)
                        w = _vertex_area_weights(v_cpu, faces_cpu).numpy().astype(np.float64, copy=False)  # (N,)
                        w_tp = float(w[np.logical_and(p, y)].sum())
                        w_fp = float(w[np.logical_and(p, ~y)].sum())
                        w_fn = float(w[np.logical_and(~p, y)].sum())
                        w_tn = float(w[np.logical_and(~p, ~y)].sum())
                        micro_w.add(w_tp, w_fp, w_fn, w_tn)
                        # per-sample weighted summary (not used for macro by default)
                        w_metrics = _metrics_from_confusion_f(w_tp, w_fp, w_fn, w_tn)
                    except Exception:
                        w_metrics = None

                per_sample_rows.append(
                    {
                        "sample_id": sid,
                        "category": cat,
                        "threshold": float(chosen_threshold),
                        "precision": m["precision"],
                        "recall": m["recall"],
                        "f1": m["f1"],
                        "iou": m["iou"],
                        "accuracy": m["accuracy"],
                        "gt_pos_ratio": m["gt_pos_ratio"],
                        "pred_pos_ratio": m["pred_pos_ratio"],
                        "w_precision": (w_metrics["precision"] if w_metrics else ""),
                        "w_recall": (w_metrics["recall"] if w_metrics else ""),
                        "w_f1": (w_metrics["f1"] if w_metrics else ""),
                        "w_iou": (w_metrics["iou"] if w_metrics else ""),
                    }
                )

                # TODO(part-metrics): 增加分部位指标（例如手/足/躯干），需要稳定的 vertex index 分区映射。

    # Macro summary (mean over samples)
    def _macro_mean(key: str) -> float:
        if not macro_metrics:
            return 0.0
        return float(np.mean([float(m.get(key, 0.0)) for m in macro_metrics]))

    micro_m = micro.metrics()
    micro_w_m = micro_w.metrics() if faces_cpu is not None else None

    by_cat_out: Dict[str, Any] = {}
    for cat, agg in sorted(by_cat.items(), key=lambda x: x[0]):
        by_cat_out[cat] = agg.metrics()

    summary_out = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "checkpoint_meta": {
            "epoch": (ckpt_meta.get("epoch") if isinstance(ckpt_meta, dict) else None),
            "best_val_loss": (ckpt_meta.get("best_val_loss") if isinstance(ckpt_meta, dict) else None),
        },
        "data_root": data_root,
        "num_samples": int(len(dataset)),
        "threshold": float(chosen_threshold),
        "default_beta": float(beta_default),
        "recommended_threshold": best_default,
        "global": {
            "micro": micro_m,
            "macro": {
                "precision": _macro_mean("precision"),
                "recall": _macro_mean("recall"),
                "f1": _macro_mean("f1"),
                "iou": _macro_mean("iou"),
                "accuracy": _macro_mean("accuracy"),
            },
            "micro_area_weighted": micro_w_m,
            "approx": {
                "brier": float(brier),
                "pr_auc": float(pr_auc),
                "roc_auc": float(roc_auc),
            },
        },
        "by_category_micro": by_cat_out,
        "by_category_samples": by_cat_samples,
        "notes": [
            "macro 为逐样本均值；micro 为全顶点混合统计。",
            "area-weighted 指标为近似（把三角形面积平均分到三个顶点），用于缓解顶点密度不均带来的偏差。",
            "TODO: 分部位指标（手/足等）尚未实现。",
        ],
    }
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2, ensure_ascii=False)

    # Write per-sample CSV
    csv_path = os.path.join(out_dir, "metrics_per_sample.csv")
    fieldnames = [
        "sample_id",
        "category",
        "threshold",
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
        "gt_pos_ratio",
        "pred_pos_ratio",
        "w_precision",
        "w_recall",
        "w_f1",
        "w_iou",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in per_sample_rows:
            w.writerow(row)

    # Save plots (key evaluation visualizations)
    plot_paths = _save_plots(
        out_dir=out_dir,
        thresholds=thresholds,
        metrics_curve=metrics_curve,
        best_by_beta=best_by_beta,
        best_default=best_default,
        beta_default=float(beta_default),
        by_category_micro=by_cat_out,
        by_category_samples=by_cat_samples,
    )

    print("\n" + "=" * 70)
    print("Evaluation done.")
    print(f"- threshold_sweep.json: {os.path.join(out_dir, 'threshold_sweep.json')}")
    print(f"- metrics_summary.json: {os.path.join(out_dir, 'metrics_summary.json')}")
    print(f"- metrics_per_sample.csv: {csv_path}")
    for p in plot_paths:
        print(f"- plot: {p}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

