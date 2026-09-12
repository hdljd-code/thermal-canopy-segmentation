import argparse
import tempfile
import sys
from pathlib import Path


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if sys.path[0] != PROJECT_ROOT:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image

from scripts.evaluate import get_prediction, load_model, save_gray_prediction


MODEL_ENTRIES = ("baseline", "msb", "asf", "asf_msb")


def predict_single_image(model_info, image_path, output_path):
    image_path = Path(image_path)
    output_path = Path(output_path)

    if not image_path.is_file():
        raise FileNotFoundError("未找到输入图片: {}".format(image_path))
    if output_path.exists():
        raise FileExistsError("输出文件已存在: {}".format(output_path))

    image = Image.open(str(image_path)).convert("RGB")
    prediction = get_prediction(model_info, image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_gray_prediction(prediction, str(output_path))
    print("[Predict] 输出文件: {}".format(output_path))


class _SelfCheckNet(torch.nn.Module):
    def forward(self, images):
        batch_size, _, height, width = images.shape
        logits = torch.zeros(
            (batch_size, 2, height, width),
            dtype=images.dtype,
            device=images.device,
        )
        logits[:, 1, :, width // 2:] = 1.0
        return logits


def self_check():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        image_path = temp_root / "input.png"
        output_path = temp_root / "prediction.png"

        image_array = np.zeros((16, 24, 3), dtype=np.uint8)
        Image.fromarray(image_array, mode="RGB").save(str(image_path))

        model_info = {
            "net": _SelfCheckNet().eval(),
            "device": torch.device("cpu"),
            "input_shape": (32, 32),
            "cuda": False,
            "num_classes": 2,
        }
        predict_single_image(model_info, image_path, output_path)

        prediction = np.array(Image.open(str(output_path)), dtype=np.uint8)
        assert prediction.shape == image_array.shape[:2]
        assert set(np.unique(prediction)).issubset({0, 1})

    print("[SingleImageInference] PASS")


def main():
    parser = argparse.ArgumentParser(description="Single-image segmentation inference")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--ckpt", type=str, help="Model checkpoint path")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/prediction.png",
        help="Output binary mask path",
    )
    parser.add_argument(
        "--model-entry",
        type=str,
        default="baseline",
        choices=MODEL_ENTRIES,
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=2,
        default=[512, 512],
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a synthetic CPU self-check without project data or weights",
    )
    args = parser.parse_args()

    if args.self_check:
        self_check()
        return

    if not args.image or not args.ckpt:
        parser.error("Inference requires both --image and --ckpt")

    model_info = load_model(
        num_classes=args.num_classes,
        model_path=args.ckpt,
        input_shape=tuple(args.input_shape),
        cuda=not args.no_cuda,
        model_entry=args.model_entry,
    )
    predict_single_image(model_info, args.image, args.output)


if __name__ == "__main__":
    main()
