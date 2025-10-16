import torch
import torch.nn as nn

class DeepPerceptron(nn.Module):
    """
    多层感知机（深度感知机）——适用于一维长度为 input_dim 的输入。
    参数:
        input_dim (int): 输入特征维度（例如 20）。
        hidden_sizes (list[int]): 隐层单元数列表，例如 [64, 32, 16]。
        num_classes (int): 输出类别数。
        dropout (float): 每个隐层后的 dropout 比例（0.0 表示不使用）。
        use_bn (bool): 是否在隐层后使用 BatchNorm1d。
    前向输出为 raw logits（适用于 nn.CrossEntropyLoss）。
    """
    def __init__(self,
                 input_dim = 20,
                 hidden_sizes =  [64, 32, 16],
                 num_classes = 4,
                 dropout: float = 0.3,
                 use_bn: bool = True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))  # 输出层（logits）
        self.net = nn.Sequential(*layers)

        # 权重初始化（线性层使用 Kaiming 初始化，bias 置零）
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor(shape=(B, input_dim)), dtype=float32
        return: Tensor(shape=(B, num_classes)), raw logits
        """
        return self.net(x)
