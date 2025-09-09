# -*- coding: utf-8 -*-
# @Time    : 9/9/25
# @Author  : Yaojie Shen
# @Project : adamw-bf16
# @File    : test_optimizer.py

import pytest
import torch

from adamw_bf16 import AdamWBF16


@pytest.mark.parametrize("cautious", [False, True])
def test_adamwbf16_step(cautious):
    # 创建一个简单的线性模型
    model = torch.nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    x = torch.randn(3, 4, dtype=torch.bfloat16)
    y = torch.randn(3, 2, dtype=torch.bfloat16)

    # 初始化优化器
    optimizer = AdamWBF16(model.parameters(), lr=1e-2, weight_decay=1e-2, cautious=cautious)

    # 前向
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)

    # 反向
    loss.backward()

    # 保存旧参数
    old_params = [p.clone() for p in model.parameters()]

    # 更新参数
    optimizer.step()

    # 确保参数更新
    for old_p, new_p in zip(old_params, model.parameters()):
        assert not torch.allclose(old_p, new_p), "Parameters did not update"

    # 确保梯度被清零（AdamWBF16 会在 step 内清零）
    for p in model.parameters():
        assert torch.all(p.grad == 0), "Gradients were not cleared"
