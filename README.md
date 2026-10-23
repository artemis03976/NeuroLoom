<p align="center">
  <img src="brand/assets/neuroloom-lockup-zh-light.svg" height="64" alt="NeuroLoom 织络">
</p>

NeuroLoom（织络）是一个可视化神经网络搭建工具：把 PyTorch 模型变成可编辑、带数据流动画的节点图，并在图与代码之间双向转换。完整需求见 [PRD v0.1](https://claude.ai/artifact/VfcmRSKrMMfVHtPBmHVE2R)。

**当前阶段：M0 技术验证（已通过闸口）。** 结论与遗留问题见 [docs/m0-report.md](docs/m0-report.md)。

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
python/.venv/Scripts/python -m pip install torch numpy pytest --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
# macOS / Linux
python/.venv/bin/python -m pip install torch numpy pytest --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

启动编辑器（开发服务器会自动拉起 Python 后端）：

```bash
npm run dev
```

打开 http://localhost:5173 。也可以用 `NEUROLOOM_PYTHON` 指定其他已装好 torch 的解释器。

## 常用命令

| 命令 | 作用 |
| --- | --- |
| `npm run dev` | 启动编辑器 |
| `npm test` | TypeScript 单元测试 + 图 ↔ 代码往返测试（无 torch 时跳过往返部分） |
| `npm run test:py` | Python 后端测试 |
| `npm run typecheck` | 全部工作区类型检查 |
| `npm run build` | 构建编辑器静态文件 |
| `npm run brand` | 从 `brand/palette.json` 重新生成 logo、图标与配色 token |

## 目录结构

```
packages/core     IR、节点规格、形状推断、代码生成（纯 TypeScript，无 UI 依赖）
apps/studio       编辑器界面（React + React Flow）；dev/pythonBridge.ts 连接 Python 后端
python/neuroloom  Python 后端：torch.fx 导入、JSON-RPC stdio 服务、CLI
brand             logo、配色与生成脚本，见 brand/README.md
docs              阶段报告
```

后端也可以单独当命令行用：

```bash
python -m neuroloom import model.py --input x=B,784
```
