from __future__ import annotations

import os
import sys
import json
import csv
import time
import atexit
import random
import argparse
import datetime
import traceback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


from utils.runtime import cvtColor, preprocess_input, resize_image
from utils.segmentation_metrics import compute_mIoU


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def setup_console_logger(run_dir):
    log_path = os.path.join(run_dir, "console.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    class TeeStream:
        def __init__(self, stream, file):
            self.stream   = stream
            self.file     = file
            self.encoding = getattr(stream, "encoding", "utf-8")

        def write(self, data):
            try:
                self.stream.write(data)
                self.stream.flush()
            except Exception:
                pass
            try:
                if self.file is not None and (not self.file.closed):
                    self.file.write(data)
                    self.file.flush()
            except Exception:
                pass

        def flush(self):
            try:
                self.stream.flush()
            except Exception:
                pass
            try:
                if self.file is not None and (not self.file.closed):
                    self.file.flush()
            except Exception:
                pass

        def isatty(self):
            try:
                return self.stream.isatty()
            except Exception:
                return False

    sys.stdout = TeeStream(orig_stdout, log_file)
    sys.stderr = TeeStream(orig_stderr, log_file)

    def close_log_file():
        try:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
        except Exception:
            pass
        try:
            if log_file is not None and (not log_file.closed):
                log_file.flush()
                log_file.close()
        except Exception:
            pass

    atexit.register(close_log_file)

    print(f"[INFO] run_dir: {run_dir}")
    print(f"[INFO] console log: {log_path}")
    return log_path


def parse_ckpt_args(ckpt_list):
    experiments = []
    for item in ckpt_list:
        if "=" not in item:
            raise ValueError(f"Invalid --ckpt format: expected name=path, got {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not os.path.exists(path):
            print(f"[WARN] checkpoint 不存在: {path}")
        experiments.append({"exp_name": name, "model_path": path})
    return experiments


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        tensor_count = sum(1 for v in checkpoint.values() if isinstance(v, torch.Tensor))
        if tensor_count > 0:
            return checkpoint
    raise ValueError("无法从 checkpoint 中提取有效的 state_dict")


def normalize_key_name(key):
    changed = True
    while changed:
        changed = False
        for prefix in ["module.", "net.", "model."]:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def load_weights_to_net(net, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到权重文件: {model_path}")

    print(f"[INFO] 加载权重: {model_path}")
    checkpoint      = torch.load(model_path, map_location=device)
    pretrained_dict = extract_state_dict(checkpoint)
    model_dict      = net.state_dict()

    load_keys       = []
    no_load_keys    = []
    temp_dict       = {}

    for k, v in pretrained_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        name = normalize_key_name(k)
        if name in model_dict and model_dict[name].shape == v.shape:
            temp_dict[name] = v
            load_keys.append(name)
        else:
            no_load_keys.append(name)

    print(f"[INFO] 成功加载 {len(load_keys)} 个参数")
    print(f"[INFO] 未加载 {len(no_load_keys)} 个参数")

    missing_keys = [name for name in model_dict if name not in temp_dict]
    if missing_keys:
        raise ValueError(
            "checkpoint 参数不完整: loaded={}, expected={}, missing={}".format(
                len(load_keys), len(model_dict), missing_keys[:10]
            )
        )

    net.load_state_dict(temp_dict, strict=True)
    return net


def build_unet(num_classes, pretrained, model_entry="baseline"):
    if model_entry == "baseline":
        from nets.unet import Unet as UnetClass
    else:
        from nets.model_api import get_model_class

        UnetClass = get_model_class(model_entry)

    return UnetClass(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone="resnet50",
    )


def load_model(
    num_classes, model_path, input_shape, cuda, model_entry="baseline"
):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    net = build_unet(
        num_classes=num_classes,
        pretrained=False,
        model_entry=model_entry,
    )

    net = load_weights_to_net(net, model_path, device)
    net = net.eval()

    if cuda and torch.cuda.is_available():
        net = net.cuda()

    return {
        "net": net, "device": device,
        "input_shape": input_shape,
        "cuda": cuda and torch.cuda.is_available(),
        "num_classes": num_classes
    }


def get_prediction(model_info, image):
    net         = model_info["net"]
    input_shape = model_info["input_shape"]
    cuda        = model_info["cuda"]

    image       = cvtColor(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    image_data, nw, nh  = resize_image(image, (input_shape[1],input_shape[0]))
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if cuda:
            images = images.cuda()

        pr = net(images)[0]
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

        top  = int((input_shape[0] - nh) // 2)
        left = int((input_shape[1] - nw) // 2)
        pr   = pr[top: top + nh, left: left + nw]

        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

    return Image.fromarray(np.uint8(pr), mode="L")


def save_gray_prediction(gray_pred, output_path):
    pred_np = np.array(gray_pred, dtype=np.uint8)
    pred_np = np.where(pred_np > 0, 1, 0).astype(np.uint8)
    Image.fromarray(pred_np, mode="L").save(output_path, format="PNG", compress_level=3)


def evaluate_one_checkpoint(
    exp_name, model_path, image_ids,
    num_classes, name_classes, gt_dir, jpeg_dir,
    input_shape, cuda, run_dir,
    model_entry="baseline",
):
    print("\n" + "#" * 70)
    print(f"[Eval] {exp_name}")
    print(f"[Eval] {model_path}")
    print("#" * 70)

    pred_dir = os.path.join(run_dir, exp_name, "detection-results")
    os.makedirs(pred_dir, exist_ok=True)


    model_info = load_model(
        num_classes,
        model_path,
        input_shape,
        cuda,
        model_entry=model_entry
    )


    eval_ids = image_ids
    total_time   = 0.0
    total_images = len(eval_ids)

    if cuda and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i, image_id in enumerate(tqdm(eval_ids, desc=f"Predict-{exp_name}")):
        jpg_path = os.path.join(jpeg_dir, f"{image_id}.jpg")
        png_path = os.path.join(jpeg_dir, f"{image_id}.png")
        image_path = jpg_path if os.path.exists(jpg_path) else png_path
        original_img = Image.open(image_path).convert("RGB")

        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        gray_pred = get_prediction(model_info, original_img)

        if cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        total_time += end_time - start_time

        save_gray_prediction(gray_pred, os.path.join(pred_dir, f"{image_id}.png"))

    avg_ms  = (total_time / total_images) * 1000 if total_images > 0 else 0.0
    fps     = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    gpu_mem = 0.0
    if cuda and torch.cuda.is_available():
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"[Speed] {avg_ms:.2f} ms/image, FPS={fps:.2f}, GPU_mem={gpu_mem:.1f} MB")


    first_pred = os.path.join(pred_dir, f"{eval_ids[0]}.png")
    if os.path.exists(first_pred):
        unique_vals = np.unique(np.array(Image.open(first_pred)))
        print(f"[Check] 预测图像素值: {unique_vals}")
        if not set(unique_vals).issubset({0, 1}):
            print("[WARN] 检测到异常像素值，自动归一化全部预测图")
            for image_id in eval_ids:
                p_path = os.path.join(pred_dir, f"{image_id}.png")
                if os.path.exists(p_path):
                    p_img = np.array(Image.open(p_path))
                    p_img = np.where(p_img > 0, 1, 0).astype(np.uint8)
                    Image.fromarray(p_img, mode="L").save(p_path)


    from utils.segmentation_metrics import fast_hist, per_class_iu
    per_image_rows = []
    for image_id in eval_ids:
        gt_path   = os.path.join(gt_dir, f"{image_id}.png")
        pred_path = os.path.join(pred_dir, f"{image_id}.png")
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            continue

        gt  = np.array(Image.open(gt_path))
        pr  = np.array(Image.open(pred_path))
        valid_mask = (gt != 255)
        gt_v = gt[valid_mask].flatten()
        pr_v = pr[valid_mask].flatten()

        hist_i = fast_hist(gt_v, pr_v, num_classes)
        tp_i = int(hist_i[1, 1])
        fp_i = int(hist_i[0, 1])
        fn_i = int(hist_i[1, 0])
        tn_i = int(hist_i[0, 0])
        pixel_error_i = fp_i + fn_i
        corn_iou_i = tp_i / max(tp_i + fp_i + fn_i, 1)
        iou_i = float(np.nanmean(per_class_iu(hist_i)))

        per_image_rows.append({
            "image_id": image_id,
            "TP": tp_i,
            "FP": fp_i,
            "FN": fn_i,
            "TN": tn_i,
            "pixel_error": pixel_error_i,
            "IoU": iou_i,
            "Corn_IoU": corn_iou_i
        })

    per_image_csv = os.path.join(run_dir, exp_name, "per_image_metrics.csv")
    if len(per_image_rows) > 0:
        with open(per_image_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)
        print(f"[PerImage] 已保存逐图指标: {per_image_csv}")


    hist, IoUs, PA_Recall, Precision = compute_mIoU(
        gt_dir, pred_dir, eval_ids, num_classes, name_classes
    )


    Dice_scores = np.zeros(num_classes)
    F1_scores   = np.zeros(num_classes)
    for i in range(num_classes):
        tp = hist[i, i]
        fp = np.sum(hist[:, i]) - tp
        fn = np.sum(hist[i, :]) - tp
        if (2 * tp + fp + fn) > 0:
            Dice_scores[i] = (2 * tp) / (2 * tp + fp + fn)
        if (Precision[i] + PA_Recall[i]) > 0:
            F1_scores[i] = 2 * (Precision[i] * PA_Recall[i]) / (Precision[i] + PA_Recall[i])

    mIoU       = np.nanmean(IoUs) * 100
    mPA        = np.nanmean(PA_Recall) * 100
    mPrecision = np.nanmean(Precision) * 100
    mF1        = np.nanmean(F1_scores) * 100
    mDice      = np.nanmean(Dice_scores) * 100
    OA         = np.sum(np.diag(hist)) / np.sum(hist) * 100 if np.sum(hist) > 0 else 0.0


    TN = int(hist[0, 0])
    FP = int(hist[0, 1])
    FN = int(hist[1, 0])
    TP = int(hist[1, 1])

    bg_iou      = IoUs[0] * 100
    bg_recall   = PA_Recall[0] * 100
    bg_prec     = Precision[0] * 100
    bg_f1       = F1_scores[0] * 100
    corn_iou    = IoUs[1] * 100
    corn_recall = PA_Recall[1] * 100
    corn_prec   = Precision[1] * 100
    corn_f1     = F1_scores[1] * 100


    print(f"\n[Result] {exp_name}")
    print(f"  mIoU          : {mIoU:.2f}%")
    print(f"  mPA           : {mPA:.2f}%")
    print(f"  OA            : {OA:.2f}%")
    print(f"  Corn_IoU      : {corn_iou:.2f}%")
    print(f"  Corn_F1       : {corn_f1:.2f}%")
    print(f"  Background_IoU: {bg_iou:.2f}%")
    print(f"  Background_F1 : {bg_f1:.2f}%")
    print(f"  TP={TP}  FP={FP}  FN={FN}  TN={TN}")


    result_csv = os.path.join(run_dir, exp_name, "result_table.csv")
    with open(result_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_name", "mIoU", "mPA", "mF1", "mDice", "mPrecision", "OA",
            "Background_IoU", "Background_Recall", "Background_Precision", "Background_F1",
            "Corn_IoU", "Corn_Recall", "Corn_Precision", "Corn_F1",
            "TP", "FP", "FN", "TN", "speed_ms", "fps", "gpu_mem_mb"
        ])
        writer.writerow([
            exp_name,
            f"{mIoU:.2f}", f"{mPA:.2f}", f"{mF1:.2f}", f"{mDice:.2f}",
            f"{mPrecision:.2f}", f"{OA:.2f}",
            f"{bg_iou:.2f}", f"{bg_recall:.2f}", f"{bg_prec:.2f}", f"{bg_f1:.2f}",
            f"{corn_iou:.2f}", f"{corn_recall:.2f}", f"{corn_prec:.2f}", f"{corn_f1:.2f}",
            TP, FP, FN, TN,
            f"{avg_ms:.2f}", f"{fps:.2f}", f"{gpu_mem:.2f}"
        ])


    return {
        "exp_name": exp_name,
        "weight_path": model_path,
        "weight_filename": os.path.basename(model_path),
        "mIoU": mIoU, "mPA": mPA, "mF1": mF1, "mDice": mDice,
        "mPrecision": mPrecision, "OA": OA,
        "Background_IoU": bg_iou, "Background_Recall": bg_recall,
        "Background_Precision": bg_prec, "Background_F1": bg_f1,
        "Corn_IoU": corn_iou, "Corn_Recall": corn_recall,
        "Corn_Precision": corn_prec, "Corn_F1": corn_f1,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "speed_ms": avg_ms, "fps": fps, "gpu_mem_mb": gpu_mem,
        "pred_dir": pred_dir,
        "per_image_csv": per_image_csv if len(per_image_rows) > 0 else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Model evaluation entry point")


    parser.add_argument("--split-file", type=str, default="dataset/splits/test.txt")
    parser.add_argument("--mask-dir", type=str, default="dataset/masks")
    parser.add_argument("--image-dir", type=str, default="dataset/images")
    parser.add_argument("--input-shape",   type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num-classes",   type=int, default=2)
    parser.add_argument("--name-classes",  type=str, default="background,corn")
    parser.add_argument("--model-entry",   type=str, default="baseline",
                        choices=["baseline", "msb", "asf", "asf_msb"])

    parser.add_argument("--ckpt",          action="append", required=True,
                        help="Checkpoint in name=path format; may be repeated")


    parser.add_argument("--output-dir",    type=str, default="outputs/evaluation")


    parser.add_argument("--cuda",          action="store_true", default=True)
    parser.add_argument("--no-cuda",       action="store_true")


    parser.add_argument("--seed",          type=int, default=11)

    args = parser.parse_args()


    seed        = args.seed
    num_classes = args.num_classes
    input_shape = tuple(args.input_shape)
    name_classes = args.name_classes.split(",")
    cuda        = args.cuda and not args.no_cuda
    if cuda and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动回退 CPU")
        cuda = False


    experiments = parse_ckpt_args(args.ckpt)
    if len(experiments) == 0:
        print("[ERROR] 未指定任何 checkpoint")
        return


    split_txt = os.path.abspath(args.split_file)
    gt_dir = os.path.abspath(args.mask_dir)
    jpeg_dir = os.path.abspath(args.image_dir)


    check_paths = [
        (split_txt, "split txt"),
        (gt_dir, "GT 目录"),
        (jpeg_dir, "JPEG 目录"),
    ]
    for path, desc in check_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {desc}: {path}")

    with open(split_txt, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f if line.strip()]

    if len(image_ids) == 0:
        raise ValueError(f"split 列表为空: {split_txt}")


    timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    run_dir   = os.path.join(args.output_dir, f"eval_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    setup_console_logger(run_dir)


    args.run_dir = run_dir
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)


    seed_everything(seed)

    print("=" * 70)
    print("论文模型评价入口")
    print("=" * 70)
    print(f"  样本数量 : {len(image_ids)}")
    print(f"  GT 目录  : {gt_dir}")
    print(f"  JPEG 目录: {jpeg_dir}")
    print(f"  模型入口 : {args.model_entry}")
    print(f"  输入尺寸 : {input_shape}")
    print(f"  CUDA     : {cuda}")
    print(f"  实验数量 : {len(experiments)}")
    for exp in experiments:
        print(f"    - {exp['exp_name']}: {exp['model_path']}")
    print("=" * 70)


    all_results = []

    for exp in experiments:
        try:
            result = evaluate_one_checkpoint(
                exp_name     = exp["exp_name"],
                model_path   = exp["model_path"],
                image_ids    = image_ids,
                num_classes  = num_classes,
                name_classes = name_classes,
                gt_dir       = gt_dir,
                jpeg_dir     = jpeg_dir,
                input_shape  = input_shape,
                cuda         = cuda,
                run_dir      = run_dir,
                model_entry  = args.model_entry,
            )
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"[ERROR] 评估 {exp['exp_name']} 失败: {e}")
            traceback.print_exc()


    if len(all_results) == 0:
        raise RuntimeError("没有成功评估的 checkpoint")


    csv_fields = [
        "exp_name", "weight_path", "weight_filename",
        "model_entry",
        "mIoU", "mPA", "mF1", "mDice", "mPrecision", "OA",
        "Background_IoU", "Background_Recall", "Background_Precision", "Background_F1",
        "Corn_IoU", "Corn_Recall", "Corn_Precision", "Corn_F1",
        "TP", "FP", "FN", "TN", "speed_ms", "fps", "gpu_mem_mb", "pred_dir", "per_image_csv"
    ]

    csv_path = os.path.join(run_dir, "summary_metrics.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            row = {k: r.get(k, "") for k in csv_fields}
            row["model_entry"]  = args.model_entry
            writer.writerow(row)


    json_path = os.path.join(run_dir, "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_entry": args.model_entry,
            "num_images": len(image_ids),
            "results": all_results
        }, f, indent=2, ensure_ascii=False)


    md_path = os.path.join(run_dir, "summary_metrics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 评估汇总\n\n")
        f.write(f"- 样本数量: {len(image_ids)}\n")
        f.write(f"- 模型入口: {args.model_entry}\n\n")

        f.write("| exp_name | mIoU | mPA | OA | Corn_IoU | Corn_F1 | TP | FP | FN | TN |\n")
        f.write("|----------|-----:|----:|---:|---------:|--------:|---:|---:|---:|---:|\n")
        for r in all_results:
            f.write(
                f"| {r['exp_name']} "
                f"| {r['mIoU']:.2f} "
                f"| {r['mPA']:.2f} "
                f"| {r['OA']:.2f} "
                f"| {r['Corn_IoU']:.2f} "
                f"| {r['Corn_F1']:.2f} "
                f"| {r['TP']} "
                f"| {r['FP']} "
                f"| {r['FN']} "
                f"| {r['TN']} |\n"
            )


    print("\n" + "=" * 100)
    print("汇总结果")
    print("=" * 100)
    print(f"{'exp_name':<25} {'mIoU':>8} {'mPA':>8} {'OA':>8} {'Corn_IoU':>10} {'Corn_F1':>10}")
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['exp_name']:<25} "
            f"{r['mIoU']:>7.2f}% "
            f"{r['mPA']:>7.2f}% "
            f"{r['OA']:>7.2f}% "
            f"{r['Corn_IoU']:>9.2f}% "
            f"{r['Corn_F1']:>9.2f}%"
        )
    print("=" * 100)
    print(f"\n[INFO] 汇总文件:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")
    print(f"  run_dir: {run_dir}")


if __name__ == "__main__":
    main()
