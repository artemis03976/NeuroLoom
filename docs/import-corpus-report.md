# 导入语料库报告

分支 `v1/import-corpus`，基于 main（d4e68fb）。对应 PRD 第 13 节的指标：

> 开源模型文件静态导入成功率 ≥ 80%（自建 50 个常见仓库模型文件语料库）

**结论：语料库已收录，第一次测量是 20/50（40%），离 80% 还差 20 个文件。**
- 成功的 20 个文件中，4 个没有代码块；留出集 12 个文件中成功 6 个。
- 失败全部出在导入器：测试环境能构造并运行全部 62 个原模型。
- 失败原因分为五类，见第 8 节。其中三类合计 20 个文件，改起来不涉及 IR：
  - `forward()` 的默认参数；
  - `__init__` 的 `*args` / `**kwargs`；
  - 生成的代码丢了模块级函数。

方案在收录前经过确认。收录时又按实际运行的结果改了几处，见第 6 节。

## 1. 成功的定义

导入器只读一个文件，入口类由清单指定。一个文件同时满足以下三条才算成功：

1. **能导入**：静态导入没有报错。
2. **等价**：
   - 按仓库自己的方式构造原模型，再用同样的方式构造生成代码中的模型（生成的代码放在原文件旁边运行）；
   - 两者的 `state_dict` 键名和形状一致，能用 `strict=True` 加载原模型的权重；
   - eval 模式下，同一输入的输出一致（`atol=1e-5`、`rtol=1e-4`）。
3. **没有整类降级**：入口类以及它用到的类，都不是 L3（整类保留源码）。代码块是允许的。

结果怎么报：
- **指标**：50 个测量文件中成功的个数。
- **另外报两项**：
  - 严格口径：没有任何代码块；
  - 每个文件中各类的级别和代码块数。
- **留出集的 12 个文件单独报告**。修改导入器时不看这 12 个文件。

## 2. 选取规则

### 候选池

- **搜索**：2026-10-24 执行了 39 条固定的 GitHub 搜索，条件是 Python、≥ 1,000 star、按 star 排序；结果按 7 个类别合并。
- **补充**：另加 huggingface/transformers，因为这些搜索没有命中它。
- **归属**：一个仓库只归入一个类别；同时出现在几个类别里的，按“分割/检测 → 生成 → 序列 → ViT → 语言模型 → CNN → 教程”的顺序归入第一个。

### 走法

- 在每个类别的池子里，按 star 从高到低逐个判断是否合格。
- 前 N 个合格的文件进测量集，再往后 2 个进留出集。
- 配额按文件计。

### 每个仓库取哪个文件

- 一般取一个文件：README 主推模型的顶层 `nn.Module` 所在的文件。
- 模型合集取 README 列出的第一个模型。
- 大型库（timm、transformers、diffusers、pytorch/examples、pytorch/tutorials）在所属类别中最多取 2 个文件。

### 不合格的原因

| 代码 | 含义 |
| --- | --- |
| `NOT_MODEL` | 仓库的主体不是模型定义：应用、框架、工具包、可视化、数据集、列表，或只是调用别的库 |
| `LICENSE` | 固定提交中的 LICENSE 文件不是 MIT、BSD-2/3、Apache-2.0、ISC、0BSD、MIT-0、Unlicense；或者没有 LICENSE；或者附加了限制（非商用、OpenRAIL、自定义条款）。以文件内容为准 |
| `NOT_TORCH` | JAX、TensorFlow、MLX、Paddle 等 |
| `NATIVE` | 构造模型或在 CPU 上前向需要编译扩展或只能在 GPU 上运行的代码 |
| `DEPS` | Python 导入这个文件时执行的代码（包括所在包的 `__init__.py`）需要允许范围以外的包 |
| `BROKEN` | 按仓库的写法，在测试环境里构造不了 |
| `NOTEBOOK` | 模型只在 .ipynb 里 |
| `INPUT` | `forward()` 的输入不是张量或张量的简单容器 |
| `CATEGORY` | 明显属于别的类别（不会移过去） |
| `DUPLICATE` | 已选文件的分叉或副本 |
| `UNAVAILABLE` | 克隆失败，或者 HEAD 里没有模型代码 |

**依赖的允许范围**：
- 测试环境要装的包：torch、torchvision、torchaudio、numpy、scipy、einops、timm、transformers、diffusers；
- 安装这些包时自带的依赖：Pillow、huggingface_hub、safetensors、PyYAML、tqdm、requests、packaging 等。
- 只在函数里导入的，或用 `try/except ImportError` 包住的可选依赖，不算。

## 3. 类别配额

| 类别 | 测量集 | 留出集 | 走过的仓库 |
| --- | ---: | ---: | ---: |
| CNN 分类 / 骨干网络 | 8 | 2 | 30 |
| 视觉 Transformer | 7 | 1（池子用完） | 44 |
| 语言模型 / NLP | 9 | 2 | 95 |
| 生成模型（GAN、VAE、扩散） | 8 | 2 | 64 |
| 分割 / 检测 | 6 | 2 | 20 |
| 序列、音频、时间序列 | 6 | 1（池子用完） | 31 |
| 教程和课程代码 | 6 | 2 | 13 |
| 合计 | **50** | **12** | 297 |

走过的每个仓库、判断结果和原因，都记在 `python/tests/corpus/selection.json` 里。

## 4. 收录的文件

- 文件按固定提交原样收录在 `python/tests/corpus/third_party/<id>/`，连同 Python 导入它时执行的同仓库文件，以及原 LICENSE。共 62 个条目、407 个文件，约 5 MB。
- 提交：
  - 库文件是对应安装版本的发布标签：timm v1.0.30、transformers v5.20.2、diffusers v0.42.0；
  - 其余文件是 2026-10-24 各仓库的 HEAD。
- “★”列是搜索时的 star 数。
- † 表示按统一的依赖规则补进来的文件。

### CNN 分类 / 骨干网络

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 1 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 37,236 | `timm/models/resnet.py` | `ResNet` | Apache-2.0 | `0df212b3` |
| 2 | [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) | 8,219 | `efficientnet_pytorch/model.py` | `EfficientNet` | Apache-2.0 | `7e8b0d31` |
| 3 | [kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) | 6,447 | `models/vgg.py` | `VGG` | MIT | `49b7aa97` |
| 4 | [kenshohara/3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) | 4,037 | `models/resnet.py` | `ResNet` | MIT | `540a0ea1` |
| 5 | [quark0/darts](https://github.com/quark0/darts) | 3,997 | `cnn/model.py` | `NetworkCIFAR` | Apache-2.0 | `f276dd34` |
| 6 | [xiaolai-sqlai/mobilenetv3](https://github.com/xiaolai-sqlai/mobilenetv3) | 1,870 | `mobilenetv3.py` | `MobileNetV3_Small` | MIT | `1a708835` |
| 7 | [rwightman/gen-efficientnet-pytorch](https://github.com/rwightman/gen-efficientnet-pytorch) | 1,577 | `geffnet/gen_efficientnet.py` | `GenEfficientNet` | Apache-2.0 | `771ce082` |
| 8 | [kumar-shridhar/PyTorch-BayesianCNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN) | 1,574 | `models/BayesianModels/BayesianAlexNet.py` | `BBBAlexNet` | MIT | `d93bad54` |
| 留出 | [filipradenovic/cnnimageretrieval-pytorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch) † | 1,501 | `cirtorch/networks/imageretrievalnet.py` | `ImageRetrievalNet` | MIT | `1e66a417` |
| 留出 | [tonylins/pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2) | 1,405 | `MobileNetV2.py` | `MobileNetV2` | Apache-2.0 | `99f21365` |

### 视觉 Transformer

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 9 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 37,236 | `timm/models/vision_transformer.py` | `VisionTransformer` | Apache-2.0 | `0df212b3` |
| 10 | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | 37,236 | `timm/models/swin_transformer.py` | `SwinTransformer` | Apache-2.0 | `0df212b3` |
| 11 | [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) | 25,550 | `vit_pytorch/vit.py` | `ViT` | MIT | `86c8b9af` |
| 12 | [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer) | 16,094 | `models/swin_transformer.py` | `SwinTransformer` | MIT | `f82860bf` |
| 13 | [facebookresearch/dino](https://github.com/facebookresearch/dino) | 7,608 | `vision_transformer.py` | `VisionTransformer` | Apache-2.0 | `7c446df5` |
| 14 | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | 5,617 | `models/network_swinir.py` | `SwinIR` | Apache-2.0 | `6545850f` |
| 15 | [baaivision/EVA](https://github.com/baaivision/EVA) | 2,697 | `EVA-01/eva/modeling_pretrain.py` | `VisionTransformerForMaskedImageModeling` | MIT | `5623d08b` |
| 留出 | [OpenGVLab/InternVideo](https://github.com/OpenGVLab/InternVideo) | 2,412 | `InternVideo1/Pretrain/VideoMAE/modeling_pretrain.py` | `PretrainVisionTransformer` | Apache-2.0 | `3965eef1` |

### 语言模型 / NLP

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 16 | [huggingface/transformers](https://github.com/huggingface/transformers) | 168,052 | `src/transformers/models/bert/modeling_bert.py` | `BertModel` | Apache-2.0 | `18329f10` |
| 17 | [huggingface/transformers](https://github.com/huggingface/transformers) | 168,052 | `src/transformers/models/llama/modeling_llama.py` | `LlamaForCausalLM` | Apache-2.0 | `18329f10` |
| 18 | [jingyaogong/minimind](https://github.com/jingyaogong/minimind) † | 64,324 | `model/model_minimind.py` | `MiniMindForCausalLM` | Apache-2.0 | `f659b557` |
| 19 | [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 64,039 | `model.py` | `GPT` | MIT | `3adf61e1` |
| 20 | [karpathy/minGPT](https://github.com/karpathy/minGPT) | 24,965 | `mingpt/model.py` | `GPT` | MIT | `37baab71` |
| 21 | [openai/gpt-oss](https://github.com/openai/gpt-oss) † | 20,490 | `gpt_oss/torch/model.py` | `Transformer` | Apache-2.0 | `5681a6b6` |
| 22 | [jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) | 9,820 | `transformer/Models.py` | `Transformer` | MIT | `132907dd` |
| 23 | [codertimo/BERT-pytorch](https://github.com/codertimo/BERT-pytorch) | 6,531 | `bert_pytorch/model/bert.py` | `BERT` | Apache-2.0 | `d10dc4f9` |
| 24 | [649453932/Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) | 5,731 | `models/TextCNN.py` | `Model` | MIT | `6cb26819` |
| 留出 | [huggingface/pytorch-openai-transformer-lm](https://github.com/huggingface/pytorch-openai-transformer-lm) | 1,526 | `model_pytorch.py` | `TransformerModel` | MIT | `bfd8e098` |
| 留出 | [graykode/gpt-2-Pytorch](https://github.com/graykode/gpt-2-Pytorch) | 1,012 | `GPT2/model.py` | `GPT2LMHeadModel` | MIT | `401078fd` |

### 生成模型（GAN、VAE、扩散）

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 25 | [huggingface/diffusers](https://github.com/huggingface/diffusers) | 34,787 | `src/diffusers/models/unets/unet_2d_condition.py` | `UNet2DConditionModel` | Apache-2.0 | `95f9b892` |
| 26 | [huggingface/diffusers](https://github.com/huggingface/diffusers) | 34,787 | `src/diffusers/models/autoencoders/autoencoder_kl.py` | `AutoencoderKL` | Apache-2.0 | `95f9b892` |
| 27 | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | 25,265 | `models/networks.py` | `ResnetGenerator` | BSD-2-Clause¹ | `2a7afba2` |
| 28 | [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN) | 17,455 | `implementations/acgan/acgan.py` | `Generator` | MIT | `36d3c77e` |
| 29 | [AntixK/PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE) | 7,679 | `models/vanilla_vae.py` | `VanillaVAE` | Apache-2.0 | `a6896b94` |
| 30 | [tamarott/SinGAN](https://github.com/tamarott/SinGAN) | 3,342 | `SinGAN/models.py` | `GeneratorConcatSkip2CleanAdd` | MIT | `df38a421` |
| 31 | [ajbrock/BigGAN-PyTorch](https://github.com/ajbrock/BigGAN-PyTorch) | 2,928 | `BigGAN.py` | `Generator` | MIT | `98459431` |
| 32 | [bytetriper/RAE](https://github.com/bytetriper/RAE) † | 2,088 | `src/stage1/rae.py` | `RAE` | MIT | `a4d18c4d` |
| 留出 | [Janspiry/Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) † | 1,832 | `models/network.py` | `Network` | MIT | `136b29f5` |
| 留出 | [rosinality/vq-vae-2-pytorch](https://github.com/rosinality/vq-vae-2-pytorch) | 1,805 | `vqvae.py` | `VQVAE` | MIT² | `ef5f67c4` |

### 分割 / 检测

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 33 | [facebookresearch/detr](https://github.com/facebookresearch/detr) † | 15,351 | `models/detr.py` | `DETR` | Apache-2.0 | `29901c51` |
| 34 | [qubvel-org/segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) † | 11,779 | `segmentation_models_pytorch/decoders/unet/model.py` | `Unet` | MIT | `089c6fcb` |
| 35 | [CSAILVision/semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) | 5,082 | `mit_semseg/models/models.py` | `SegmentationModule` | BSD-3-Clause | `8f27c9b9` |
| 36 | [sgrvinod/a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) | 3,118 | `model.py` | `SSD300` | MIT | `5d5b7596` |
| 37 | [Tramac/awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch) † | 3,070 | `core/models/fcn.py` | `FCN32s` | Apache-2.0 | `d37d2a17` |
| 38 | [jfzhang95/pytorch-deeplab-xception](https://github.com/jfzhang95/pytorch-deeplab-xception) | 2,998 | `modeling/deeplab.py` | `DeepLab` | MIT | `9135e104` |
| 留出 | [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch) | 2,708 | `network/_deeplab.py` | `DeepLabV3` | MIT | `4e1087de` |
| 留出 | [fxia22/pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) | 2,364 | `pointnet/model.py` | `PointNetDenseCls` | MIT | `bafbf401` |

### 序列、音频、时间序列

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 39 | [r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder) | 2,374 | `wavenet_vocoder/wavenet.py` | `WaveNet` | MIT | `a35fff76` |
| 40 | [ndrplz/ConvLSTM_pytorch](https://github.com/ndrplz/ConvLSTM_pytorch) | 2,270 | `convlstm.py` | `ConvLSTM` | MIT | `9f662ba2` |
| 41 | [salesforce/awd-lstm-lm](https://github.com/salesforce/awd-lstm-lm) | 1,990 | `model.py` | `RNNModel` | BSD-3-Clause | `1d466ec5` |
| 42 | [r9y9/deepvoice3_pytorch](https://github.com/r9y9/deepvoice3_pytorch) | 1,977 | `deepvoice3_pytorch/__init__.py` | `MultiSpeakerTTSModel` | MIT | `f90255c9` |
| 43 | [dalinvip/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch](https://github.com/dalinvip/cnn-lstm-bilstm-deepcnn-clstm-in-pytorch) | 1,450 | `models/model_BiGRU.py` | `BiGRU` | Apache-2.0 | `afbbc4dd` |
| 44 | [chickenbestlover/RNN-Time-series-Anomaly-Detection](https://github.com/chickenbestlover/RNN-Time-series-Anomaly-Detection) | 1,303 | `model/model.py` | `RNNPredictor` | Apache-2.0 | `21ddb78f` |
| 留出 | [sooftware/conformer](https://github.com/sooftware/conformer) | 1,133 | `conformer/model.py` | `Conformer` | Apache-2.0 | `47863edd` |

### 教程和课程代码

| # | 仓库 | ★ | 文件 | 入口类 | 许可证 | 提交 |
| --- | --- | ---: | --- | --- | --- | --- |
| 45 | [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) | 32,515 | `tutorials/01-basics/feedforward_neural_network/main.py` | `NeuralNet` | MIT | `0500d3df` |
| 46 | [pytorch/examples](https://github.com/pytorch/examples) | 24,075 | `mnist/main.py` | `Net` | BSD-3-Clause | `acc295dc` |
| 47 | [pytorch/examples](https://github.com/pytorch/examples) | 24,075 | `word_language_model/model.py` | `RNNModel` | BSD-3-Clause | `acc295dc` |
| 48 | [pytorch/tutorials](https://github.com/pytorch/tutorials) | 9,383 | `beginner_source/basics/quickstart_tutorial.py` | `NeuralNetwork` | BSD-3-Clause | `c8ccbbf7` |
| 49 | [pytorch/tutorials](https://github.com/pytorch/tutorials) | 9,383 | `beginner_source/blitz/neural_networks_tutorial.py` | `Net` | BSD-3-Clause | `c8ccbbf7` |
| 50 | [jcjohnson/pytorch-examples](https://github.com/jcjohnson/pytorch-examples) | 4,917 | `nn/two_layer_net_module.py` | `TwoLayerNet` | MIT | `0003fcdc` |
| 留出 | [cs230-stanford/cs230-code-examples](https://github.com/cs230-stanford/cs230-code-examples) | 4,232 | `pytorch/vision/model/net.py` | `Net` | MIT | `478e747b` |
| 留出 | [sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) | 2,892 | `models.py` | `DecoderWithAttention` | MIT | `cc9c7e2f` |

¹ 仓库自己的代码是 BSD-2-Clause，LICENSE 里另附 pix2pix 和 DCGAN 代码的 BSD 许可。
² 仓库自己的代码是 MIT，LICENSE 里另附借用代码的 MIT 和 Apache-2.0 许可。

## 5. 跳过的仓库

| 类别 | NOT_MODEL | LICENSE | DEPS | NATIVE | 其他 |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN | 8 | 5 | 3 | 0 | CATEGORY 1、DUPLICATE 1、INPUT 1、UNAVAILABLE 1 |
| ViT | 8 | 7 | 5 | 0 | CATEGORY 14（多是名字含 “vit” 的 VITS 语音模型）、NOT_TORCH 2、DUPLICATE 1 |
| 语言模型 | 53 | 3 | 15 | 2 | NOT_TORCH 5、CATEGORY 4、BROKEN 2、INPUT 1 |
| 生成模型 | 26 | 8 | 14 | 2 | NOT_TORCH 4、CATEGORY 1 |
| 分割 / 检测 | 2 | 5 | 3 | 2 | — |
| 序列 | 7 | 4 | 5 | 0 | NOT_TORCH 7、CATEGORY 1 |
| 教程 | 1 | 5 | 1 | 0 | — |

较有名的几个：
- **LICENSE**：
  - ultralytics 的 YOLO 系列是 AGPL-3.0，milesial/Pytorch-UNet 是 GPL-3.0；
  - DiT、flow_matching、VRT 是 CC BY-NC 4.0，stylegan3、stylegan2-ada、VoxFormer、MambaVision 用 NVIDIA 的非商用许可；
  - 中文教程仓库多数没有 LICENSE 文件。
- **DEPS**：
  - lucidrains 的大部分仓库；
  - open_clip 和 TinyCLIP（ftfy）、Cadene/pretrained-models.pytorch（six）；
  - LaTeX-OCR（x_transformers）；
  - ViTPose、ViT-Adapter（mmcv）；
  - gpt-neox（deepspeed）、BioGPT 和 ONE-PEACE（fairseq）；
  - deepspeech.pytorch、mini-omni（lightning）。
- **NATIVE**：
  - maskrcnn-benchmark、Deformable-DETR（编译的 RoI / 可变形注意力算子）；
  - RWKV-LM（导入时编译 CUDA 内核）；
  - rosinality/stylegan2-pytorch。

## 6. 收录时对方案的改动

- **依赖检查改为跟随包的 `__init__.py`**：
  - 草稿阶段只沿文件自己的 import 检查依赖。收录时按 Python 实际执行的文件重查（导入 `a.b.c` 会先执行 `a/__init__.py`、`a/b/__init__.py`），发现：
    - Cadene/pretrained-models.pytorch 的包导入了 `six`；
    - open_clip 的包经由 tokenizer 导入了 `ftfy`；
    - TinyCLIP 是 open_clip 的分叉，同样需要 `ftfy`。之前它被记为重复，现在改为 DEPS。
  - 按走法顺延：PyTorch-BayesianCNN 升入测量集，tonylins/pytorch-mobilenet-v2 进留出集，TextCNN 回到测量集。
- **新增 `BROKEN`**：
  - BERT-NER-Pytorch 用 `init_weights()` 初始化权重，transformers 5 上构造失败；
  - llm-architecture-from-scratch 先给 `self.mask` 赋值，又 `register_buffer('mask')`。
  - 语言模型的留出集因此改为 pytorch-openai-transformer-lm 和 graykode/gpt-2-Pytorch。
- **库文件改为固定在发布标签**：
  - timm、transformers、diffusers 的 7 个文件，改为固定在与安装版本对应的发布标签，文件与安装包里的完全相同；原模型直接从安装的包运行。
  - `requirements-dev.txt` 固定了这三个版本。如果装的版本不同，测试会明确报错。
- **deepvoice3**：
  - 补 `version.py` 后能导入。但用 mel 目标做 teacher forcing 时，它在当前的 PyTorch 上对非连续张量调用 `view` 会出错，所以改用推理模式：不给 mel 目标，解码最多 8 步。
  - 仓库的 `builder.deepvoice3()` 总是构造原文件里的类。所以构造时改为用该文件自己的 `MultiSpeakerTTSModel` 和 `AttentionSeq2Seq`，把 builder 造出的编码器、解码器、转换器组装起来；否则比较的是原模型和它自己。

## 7. 测试环境

以下都在测试代码里处理，收录的文件不改：
- **离线**：
  - 网络连接被关闭；
  - 构造时下载预训练权重（torch.hub、torchvision）返回空，模型保持随机初始化；
  - Hugging Face 设为离线。RAE 需要的 DINOv2 编码器和 ViT-MAE 解码器配置，在临时目录里用 `save_pretrained` 生成小模型。
- **脚本式文件**（4 个教程和 PyTorch-GAN 的 ACGAN）：只执行 import、函数和类定义，以及只读取前面已定义的名字、不调用函数的赋值。下载数据和训练的代码不执行。
- **隔离**：每个文件在自己的仓库目录下导入，结束后清掉它的模块和对 `sys.path` 的改动，避免 `models`、`utils` 这类同名包互相干扰。
- **安装时生成的文件**：deepvoice3 的 `version` 模块用一个只含 `__version__` 的模块代替。
- **耗时**：全部 62 个原模型在本机构造并运行一次约 9 秒，整个测试约 20 秒。

## 8. 第一次测量

| 类别 | 测量集成功 | 留出集成功 |
| --- | ---: | ---: |
| CNN | 5/8 | 1/2 |
| ViT | 0/7 | 0/1 |
| 语言模型 | 3/9 | 1/2 |
| 生成模型 | 3/8 | 1/2 |
| 分割 / 检测 | 2/6 | 0/2 |
| 序列 | 1/6 | 1/1 |
| 教程 | 6/6 | 2/2 |
| 合计 | **20/50（40%）** | **6/12** |

成功的 20 个中，只有 4 个教程文件没有代码块，其余都含有代码块。

测量集中失败的 30 个文件，按原因分为五类：

| 原因 | 文件 | 个数 |
| --- | --- | ---: |
| 入口类继承自定义基类或库基类，而不是 `nn.Module`：要么找不到入口类，要么整类保留为源码 | PyTorch-BayesianCNN、transformers 的 BERT 和 Llama、minimind、diffusers 的 UNet 和 AutoencoderKL、PyTorch-VAE、smp 的 `Unet`、CSAIL 的 `SegmentationModule` | 9 |
| `forward()` 有默认参数或可变参数，整类保留为源码 | timm 的 ViT、nanoGPT、minGPT、attention-is-all-you-need 的 Encoder / Decoder、wavenet_vocoder、ConvLSTM、awd-lstm-lm、deepvoice3、RNN 时间序列异常检测 | 9 |
| `__init__` 用了 `*args`、`**kwargs` 或仅限关键字参数，整类保留为源码 | timm 的 Swin、vit-pytorch、官方 Swin-Transformer、DINO、SwinIR、BigGAN、Tramac 的 FCN | 7 |
| 生成的代码丢了仓库构造模型用的模块级函数（`build`、`_create_model`、`get_norm_layer`，以及 timm 注册的构造函数） | timm 的 ResNet、gen-efficientnet、CycleGAN、DETR | 4 |
| 不支持 `bool` 类型的输入 | EVA-01 | 1 |

留出集中另有一类：`forward()` 中途返回（PointNet），整类保留为源码。

## 9. 下一步

按影响从大到小：

1. **`forward()` 的默认参数和可变参数（9 个）**：带默认值的参数可以作为可选端口（静态导入已经支持子模块的可选参数）。
2. **`__init__` 的 `*args` / `**kwargs` / 仅限关键字参数（7 个）**：作为超参数或原样保留。
3. **保留模块级函数（4 个）**：生成的文件应当能替代原文件。仓库用来构造模型的函数要原样保留，现在只保留代码块用到的函数。
4. **`bool` 输入（1 个）**。
5. **基类（9 个）**：把基类的 `__init__` / `forward` 合并进来，或者把库基类（`PreTrainedModel`、`ModelMixin`）当作 `nn.Module` 看待。

- 前四项都改好，最多能到 41/50。实际可能还会暴露下一层问题。
- 达到 80% 需要再成功 20 个文件。
- 修改时只看测量集，最后再用留出集检查。

## 10. 文件

- `python/tests/corpus/manifest.json`：每个文件的出处、提交、sha256、许可证、类别、入口类、构造方式、示例输入、导入时执行的同仓库文件，以及属于测量集还是留出集。
- `python/tests/corpus/selection.json`：选取过程，包括搜索条件，以及走过的每个仓库的判断和原因。
- `python/tests/corpus/fetch.py`：按提交下载文件并校验 sha256。`--update` 记录新的哈希，并重写 `THIRD_PARTY_NOTICES.md`。
- `python/tests/corpus/realworld.py`：构造原模型，并与生成的代码比较。
- `packages/core/test/realworld.test.ts`：计算指标，打印每个文件的结果和原因。下限是 20，只升不降。
- 原来的 10 个自写文件移到了 `python/tests/corpus/handwritten/`。
