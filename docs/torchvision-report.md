# torchvision 导入报告

分支 `v1/torchvision-import`，基于 main（35aec77）。对应 M3 报告“已知限制”中尚未测量的 PRD 成功指标：

> torchvision 分类模型导入成功率 ≥ 90% 无代码块降级（动态模式批量导入测试）

**结论：按 PRD 的字面口径（没有代码块）是 77/80（96%），达标；按更严格的口径是 60/80（75%），未达标。**
- 严格口径还要求：没有 L1+L3 / L3 的类，也没有因为无法追踪而整体保留为外部调用的复合模块。
- 例如 ViT 没有代码块，但 `VisionTransformer` 整个是一个外部模块，画布上只有一个方框，所以严格口径不算它。
- 改动前，严格口径约为 49%。

## 测量方法

`packages/core/test/torchvision.test.ts`，装了 torchvision 才运行（CI 已安装）：

- 对 `torchvision.models` 中全部 80 个分类模型，按用户代码的写法（`self.net = torchvision.models.resnet18()`）做动态导入，统计严格口径的比例。低于 60 个时测试失败。
- 每个结构化的模型族抽 1–2 个，共 14 个，检查导入后生成的代码：
  1. 用 `strict=True` 加载原模型的权重，输出一致（`atol=1e-5`）；
  2. 再静态导入一次，生成的代码不变（不动点）。
- 本机运行约 40 秒（80 个模型的导入约 20 秒）。峰值内存：工作集约 300 MB，提交约 1.35 GB。大模型（如 ViT-H/14、RegNet-Y 128GF）在 meta 设备上构建，不分配权重。

| 模型族 | 结构化 | 说明 |
| --- | --- | --- |
| AlexNet、VGG、ResNet、ResNeXt、Wide ResNet、SqueezeNet、MobileNet V2/V3、MNASNet、EfficientNet、EfficientNet V2、RegNet、ConvNeXt、GoogLeNet、Inception V3 | 60/60 | 全部 L2，没有代码块 |
| DenseNet ×4 | 0/4 | `_DenseLayer` 的输入是张量列表 |
| ShuffleNet V2 ×4 | 0/4 | `channel_shuffle` 用 `x.size()` 的值计算 `view` 的参数 |
| ViT ×5 | 0/5 | `x.shape[0]`、`class_token.expand(n, -1, -1)`、`x[:, 0]`；`VisionTransformer` 整体是外部模块 |
| Swin ×3、Swin V2 ×3 | 0/6 | `_get_relative_position_bias`、`_patch_merging_pad` 等库函数，注意力层是外部模块。<br>Swin V2 的 `SwinTransformer` 也无法追踪，且构造参数里有类，不能作为外部模块，调用它的语句是代码块 |
| MaxViT ×1 | 0/1 | 注意力层用 `getattr` |

## 改动

### 动态导入

- **第三方复合类先追踪**：有子模块的库类（torchvision 的 `BasicBlock`、`InvertedResidual` 等）先尝试追踪，追踪失败才作为外部模块。
- **构建顺序**：先在 meta 设备上构建；失败时只把参数和缓冲区放到 meta（RegNet 的构造函数用张量计算各层宽度）；最后才完全在 CPU 上构建。
- **没有参数和缓冲区的子模块可以调用多次**：Swin 的每个块调用两次 `StochasticDepth`。
- **`torch.nn.functional` 的分发函数**：`F.max_pool2d` 等只有 `*args, **kwargs`，改从 `F._max_pool2d` 取签名，`return_indices` 等参数的默认值才能识别。
- **外部库类按名导入**：例如 `from torchvision.ops.stochastic_depth import StochasticDepth`，并保留原类名。
- **`forward()` 没有调用的子模块**（GoogLeNet、Inception V3 在 eval 模式下的辅助分类器）也会导入，因此权重能用 `strict=True` 加载。
- **模块自己的 `nn.Parameter` 和缓冲区**（ConvNeXt 的 `layer_scale`、ViT 的 `class_token`）导入为 Parameter / Buffer 节点。

### IR 与节点

- **Parameter / Buffer 节点**（PRD 附录 20）：
  - 在 `__init__` 中写成 `self.x = nn.Parameter(torch.zeros(...))` 或 `self.register_buffer("x", ...)`，在 `forward()` 中读作 `self.x`。
  - 形状可以用超参数。Parameter 计入参数量；两者都不计入激活内存。
- **其他节点**：
  - 7 个函数式池化（`F.max_pool2d`、`F.adaptive_avg_pool2d` 等）；
  - `nn.Hardsigmoid`。
- **具名 `nn.Sequential`**：写成 `nn.Sequential(OrderedDict([("conv0", ...), ...]))`，静态导入也支持。
- **没有在 `forward()` 中使用的子模块**：给出警告（不是错误），在 `__init__` 中构建，参数照常计入。

### 往返稳定

- **静态导入的节点顺序**：节点按 `forward()` 中出现的顺序排列（以前先排 `__init__` 中的层）。并行分支在拓扑排序中打平时按这个顺序，因此重新导入生成的代码，语句顺序不变。
- **`__init__` 顺序**：
  - Parameter / Buffer 放在最前，按 `forward()` 中第一次使用的顺序；
  - 没有边的子模块放在最后。
  - 这两类节点在两种导入中的位置不同，固定顺序后生成的代码才是不动点。

### 模板

- 重新生成了 `templates.json` 和 `blocks.json`。
- ViT-Tiny 的 class token 和位置编码现在是 Parameter 节点，参数量 5,717,416，与 PyTorch 一致（以前少 38,016 个）。
- 注意力块（多头注意力、ViT-Tiny、GPT-2 Small）的语句顺序与源码一致：q、k、v 逐个分支计算，而不是按 `__init__` 的顺序交错。

## 剩下 20 个模型需要什么

| 需要的能力 | 能解决 | 合计 |
| --- | --- | --- |
| 形状值：`x.shape[0]`、`x.size()` 的结果参与整数运算，作为 `view` / `expand` 的参数 | ShuffleNet ×4（还需要 ViT 的其余两项） | |
| 张量下标 `x[:, 0]`；`nn.MultiheadAttention` 的 `need_weights` 参数 | 加上形状值：ViT ×5 | |
| 模块输入为张量列表（列表类型的端口） | DenseNet ×4 | 73/80（91%） |
| 库函数内联追踪，或者追踪时把无法追踪的语句保留为代码块 | Swin ×6、MaxViT ×1 | 前者可能达到 80/80；后者有代码块，不计入指标 |

要达到严格口径的 90%，需要做前三项：形状值、张量下标和列表输入。其中形状值要改 IR（需要整数类型的值和对应的边），是最大的一项。
