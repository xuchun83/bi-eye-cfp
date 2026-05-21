# Tests

单元测试覆盖 DOX 项目的核心契约：模型形状、PEFT 参数比例、
预处理流水线、损失函数稳定性、阈值逻辑与推理 API。所有测试均可
**在 CPU 上、不需要预训练权重**的环境下运行。

## 运行

```bash
pip install -r requirements-dev.txt
pytest -v
```

或带覆盖率：

```bash
pytest --cov=. --cov-report=term-missing
```

## 测试清单

| 文件 | 覆盖范围 |
|---|---|
| `test_model.py`               | 顶层 MODEL forward shape、参数计数、双眼交错机制、transform 流水线 |
| `test_fusion.py`              | DBFEM 输出形状、梯度通路 |
| `test_loss_and_thresholds.py` | AsymmetricLoss 数值稳定性、类别阈值字典完备性、apply_thresholds、multilabel Kappa |
| `test_preprocess.py`          | 三阶段预处理形状、可消融关断、降噪定性效果 |
| `test_inference_api.py`       | inference_single 公共 API 契约 |

## 设计原则

* **隔离真实数据 / 权重**：所有测试自行构造随机图像 / 张量；
* **CPU-only**：在 GitHub Actions 等无 GPU 环境也能跑通；
* **小模型**：测试用 ``adapter_dim=32`` 等小配置，单次测试 < 5s。
