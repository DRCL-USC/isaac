from CENet import CENet
import torch
import torch.nn as nn

def test_cenet():
    # 设置输入维度
    num_actor_obs = 615
    batch_size = 5

    # 初始化模型
    model = CENet(num_actor_obs)

    # 创建随机输入数据
    observations = torch.randn(batch_size, num_actor_obs)  # 输入观察值
    real_v = torch.randn(batch_size, 3)  # 实际速度
    real_p = torch.randn(batch_size, 3)  # 实际位置

    # 前向传播测试
    next_partial_observations = model(observations, real_v, real_p)

    # 打印结果
    print("Next partial observations shape:", next_partial_observations.shape)
    assert next_partial_observations.shape == (batch_size, num_actor_obs), \
        "Output shape mismatch!"

    # 编码测试
    v, p, z = model.encode(observations)
    print("v shape:", v.shape)
    print("p shape:", p.shape)
    print("z shape:", z.shape)

    # 检查形状是否正确
    assert v.shape == (batch_size, 3), "Velocity shape mismatch!"
    assert p.shape == (batch_size, 3), "Position shape mismatch!"
    assert z.shape == (batch_size, 16), "Latent state shape mismatch!"

# 运行测试
test_cenet()
