<p align="center">
  <img src="brand/assets/neuroloom-lockup-zh-light.svg" height="64" alt="NeuroLoom 织络">
</p>

NeuroLoom（织络）是一个可视化神经网络搭建工具：把 PyTorch 模型变成可编辑、带数据流动画的节点图，并在图与代码之间双向转换。完整需求见 [PRD v0.1](https://claude.ai/artifact/VfcmRSKrMMfVHtPBmHVE2R)。

**当前阶段：M3 动态导入与打磨。** 进展与遗留问题见 [docs/m3-report.md](docs/m3-report.md)（之前的阶段：[M2](docs/m2-report.md)、[M1](docs/m1-report.md)、[M0](docs/m0-report.md)）。

- **搭建**：
  - 84 个层与算子节点，以及 7 个可展开的复合模块（残差块、多头注意力、Transformer 块……）。
  - 参数可以写成引用模块超参数的表达式。
  - 选中节点按 Ctrl+G 打包为模块，双击进入；模块可复用，也可串联重复。
  - 检查器显示每个节点的形状公式、参数量和 FLOPs。
- **模板**：12 个，从 MLP 到 ResNet-50、U-Net、ViT、GPT-2，都附原始论文链接。
- **数据流**：
  - 边上显示张量形状，并有流动动画；选中节点时只有它的上下游在流动。
  - 前向传播回放：按拓扑顺序逐层演示，可以进入模块内部。
  - 1000 节点的图也能流畅编辑和平移。
- **分析**：参数量、FLOPs（与 PyTorch 的 FlopCounterMode 口径一致）、激活与显存估算，可按 batch 等符号取值计算。
- **导出**：
  - Python：每个模块一个类，依赖在前；代码块与保留的类原样写回。
  - 导出选项：形状注释、类型标注、嵌套调用、`nn.Sequential` 重复块、自测片段，也可以拆成每个类一个文件的包。
  - 画布可以导出为 SVG / PNG / PDF（浅色、深色、打印配色）。
- **导入**：
  - 默认静态解析，不执行代码，保留子模块、变量名和超参数表达式；无法结构化的语句变为代码块，无法结构化的类保留为源码。
  - “运行后导入”会执行一次代码，把 `_make_layer`、循环构造等静态读不懂的类按实际结构展开，层级照样保留。
- **桌面版**：Windows / macOS / Linux 安装包，双击 .loom 打开。
- **界面**：中文 / English，深浅色主题。

## 安装包

GitHub Release 提供：
- Windows：`.exe` 安装程序；
- macOS：x64 和 arm64 各一个 `.dmg`；
- Linux：`.deb` 和 AppImage。

Linux 上建议用 `.deb`：
- 从 Ubuntu 24.04 起，系统限制了 Chromium 沙箱要用的非特权用户命名空间。
- `.deb` 安装时会装上 NeuroLoom 的 AppArmor 配置（或者给 `chrome-sandbox` 设 setuid），沙箱可以正常工作。
- AppImage 做不到这一点，在这些系统上直接运行会在启动时退出。可以加 `--no-sandbox` 启动，但那会关闭渲染进程的沙箱；更好的办法是改用 `.deb`。
- 在其他发行版上，AppImage 可以正常使用。

## 快速开始

需要 Node.js 22+ 与 Python 3.10+。

```bash
npm install
```

创建后端用的 Python 环境（只需 CPU 版 torch）：

```bash
python -m venv python/.venv
```

```bash
# Windows
python/.venv/Scripts/python -m pip install -r python/requirements-dev.txt
# macOS / Linux
python/.venv/bin/python -m pip install -r python/requirements-dev.txt
```

启动桌面版（Electron，会同时启动开发服务器和 Python 后端）：

```bash
npm run desktop
```

或者只在浏览器里运行编辑器（http://localhost:5173）：

```bash
npm run dev
```

也可以用 `NEUROLOOM_PYTHON` 指定其他已装好 torch 的解释器；桌面版还可以在“文件 → Python 环境”里切换或一键创建环境。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `npm run desktop` | 启动桌面版（开发模式） |
| `npm run dev` | 在浏览器中启动编辑器 |
| `npm test` | TypeScript 单元测试 + 每个节点、每个模板、分层项目与手写代码的 PyTorch 往返测试（无 torch 时跳过往返部分） |
| `npm run test:py` | Python 后端测试 |
| `npm run coverage` | 核心包（`packages/core`）的测试覆盖率；行覆盖率低于 80% 时失败 |
| `npm run typecheck` | 全部工作区类型检查 |
| `npm run build:desktop` | 构建编辑器与 Electron 主进程 |
| `npm run smoke` | 启动构建好的桌面版做端到端冒烟测试（先运行 `build:desktop`） |
| `npm run perf` | 在构建好的桌面版中按 PRD 第 13 节测量帧率、修改延迟和内存（先运行 `build:desktop`） |
| `npm run dist` | 打当前平台的安装包，输出到 `apps/desktop/release/`；`npm run dist -- --dir` 只生成免安装目录 |
| `npm run startup` | 测打包后应用的冷启动时间：启动 5 次，每次用新的用户数据目录，到编辑器可用为止（先运行 `npm run dist -- --dir`） |
| `npm run specs` | 节点规格变动后，重新生成 Python 导入器用的 `python/neuroloom/specs.json` |
| `npm run templates -w @neuroloom/studio` | 从 `apps/studio/templates/*.py` 重新生成模板 |
| `npm run brand` | 从 `brand/palette.json` 重新生成 logo、图标与配色 token |

## 目录结构

```
packages/core            IR、84 个节点规格、表达式、按实例的形状推断、模型开销（FLOPs / 内存）、代码生成与导出选项、模块编辑、.loom 读写（纯 TypeScript）
packages/backend-bridge  Node 侧的 Python 后端客户端、解释器发现、环境创建
packages/host-api        编辑器与宿主（Electron / 浏览器）之间的接口约定
apps/studio              编辑器界面（React + React Flow）；templates/ 与 blocks/ 是模板和复合模块的 Python 源码
apps/desktop             Electron 壳：窗口、菜单、文件、图片导出、Python 进程管理、安装包配置
python/neuroloom         Python 后端：静态导入（ast）、动态导入（保留层级的 torch.fx 追踪）、形状探测、JSON-RPC stdio 服务、CLI
brand                    logo、配色与生成脚本，见 brand/README.md
docs                     阶段报告
```

后端也可以单独当命令行用（默认静态解析；`--mode dynamic` 执行一次代码并保留层级，`--mode trace` 是展平的 torch.fx 追踪）：

```bash
python -m neuroloom import model.py --input x=B,3,224,224 --input ids=B,16:int64
```

```bash
python -m neuroloom import model.py --mode dynamic --input x=B,3,224,224
```
