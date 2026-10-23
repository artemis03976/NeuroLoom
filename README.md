<p align="center">
  <img src="brand/assets/neuroloom-lockup-zh-light.svg" height="64" alt="NeuroLoom 织络">
</p>

NeuroLoom（织络）是一个可视化神经网络搭建工具：把 PyTorch 模型变成可编辑、带数据流动画的节点图，并在图与代码之间双向转换。完整需求见 [PRD v0.1](https://claude.ai/artifact/VfcmRSKrMMfVHtPBmHVE2R)。

**当前阶段：M1 编辑器基础。** 进展与遗留问题见 [docs/m1-report.md](docs/m1-report.md)（M0 见 [docs/m0-report.md](docs/m0-report.md)）。

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
| `npm test` | TypeScript 单元测试 + 每个节点、每个模板的 PyTorch 往返测试（无 torch 时跳过往返部分） |
| `npm run test:py` | Python 后端测试 |
| `npm run typecheck` | 全部工作区类型检查 |
| `npm run build:desktop` | 构建编辑器与 Electron 主进程 |
| `npm run smoke` | 启动构建好的桌面版做端到端冒烟测试（先运行 `build:desktop`） |
| `npm run specs` | 节点规格变动后，重新生成 Python 导入器用的 `python/neuroloom/specs.json` |
| `npm run templates -w @neuroloom/studio` | 从 `apps/studio/templates/*.py` 重新生成模板 |
| `npm run brand` | 从 `brand/palette.json` 重新生成 logo、图标与配色 token |

## 目录结构

```
packages/core            IR、84 个节点规格、形状推断、代码生成、.loom 读写（纯 TypeScript）
packages/backend-bridge  Node 侧的 Python 后端客户端、解释器发现、环境创建
packages/host-api        编辑器与宿主（Electron / 浏览器）之间的接口约定
apps/studio              编辑器界面（React + React Flow）
apps/desktop             Electron 壳：窗口、菜单、文件、Python 进程管理
python/neuroloom         Python 后端：torch.fx 导入、JSON-RPC stdio 服务、CLI
brand                    logo、配色与生成脚本，见 brand/README.md
docs                     阶段报告
```

后端也可以单独当命令行用：

```bash
python -m neuroloom import model.py --input x=B,3,224,224 --input ids=B,16:int64
```
