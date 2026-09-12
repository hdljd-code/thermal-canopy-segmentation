# 玉米冠层热红外图像分割

本仓库提供论文相关的部分公开实验框架，包含 ResNet50-U-Net 基线、训练与评价流程入口、单图推理以及冠层温度统计代码；改进模型通过统一模型接口展示其在训练与评价流程中的接入方式。

## 安装

测试环境为 Python 3.8.20、PyTorch 2.4.1 和 CUDA 11.8。

```powershell
pip install -r requirements.txt
```

## 使用

```powershell
# 训练公开基线
python .\scripts\train.py --experiment baseline_adamw
python .\scripts\train.py --experiment baseline_adamw --no-pretrained

# 推理代码验证
python .\scripts\predict.py --self-check

# 单张图片推理
python .\scripts\predict.py --image .\example.jpg --ckpt .\weights\model.pth --output .\outputs\prediction.png

# 数据集评价
python .\scripts\evaluate.py --model-entry baseline --ckpt baseline=.\weights\model.pth

# 冠层温度统计
python .\scripts\analyze_temperature.py --baseline-pred-dir .\predictions\baseline --proposed-pred-dir .\predictions\proposed
```

## 许可证

公开代码按 `LICENSE` 中的 MIT License 发布。
