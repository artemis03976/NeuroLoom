# M0 技术验证报告

2026-10-23 · 分支 `m0/tech-validation`

**结论：M0 闸口“往返原型跑通”已通过，可以进入 M1。** 图 → 代码 → 图在 4 个测试图上结构完全一致、重新生成的代码逐字节相同；手写代码 → 图 → 代码在 3 个样例上语义等价（`state_dict` 以 `strict=True` 加载，输出数值一致）。TypeScript 形状推断与 PyTorch 在 meta 设备上的实测形状全部吻合。

## 验证了什么

| PRD 中的 M0 目标 | 结果 | 证据 |
| --- | --- | --- |
| React Flow 画布 + 3 种节点 | 通过：Input、Output、Linear、ReLU（模块）、F.relu（函数），可拖入、连线、删除、改参 | 浏览器中手动验证 |
| 形状推断 | 通过：符号维度（`B`）、`in_features` 自动推断、错误定位且不向下游级联 | `packages/core/test/core.test.ts` |
| 代码导出 | 通过：可读、确定性输出；错误处写入 `TODO(neuroloom)` | 同上，含逐字节快照 |
| torch.fx 导入可行性 | 通过：自定义 Tracer、meta 设备实例化、ShapeProp 取实测形状 | `python/tests/test_importer.py` |
| 前后端通信 | 通过：JSON-RPC 2.0 over stdio；每次导入在独立子进程中执行，30 秒超时 | 同上 + 浏览器中导入验证 |
| 往返闸口 | 通过 | `packages/core/test/roundtrip.test.ts` |

测试总数：TypeScript 24 个（含 7 个调用真实 Python 后端的往返用例），Python 6 个，全部通过；两个工作区类型检查无错误；生产构建通过。

## 过程中发现的问题

1. **返回值顺序曾随画布位置变化（已修复）。** 导入的自编码器经自动布局后，第二个输出节点被排到第一个左边，生成的 `return` 元组顺序随之颠倒，行为改变。现在 `forward()` 的参数和返回值顺序固定为创建顺序，与布局无关；回归用例会先打乱布局再做数值等价检查。
2. **Vite 偶发提供过期的 `packages/core` 编译结果。** 对 app 根目录之外的源文件连续快速修改两次后，第二次修改未被重新转换，页面因此白屏；重启或 touch 文件即可恢复。M1 如果复现频繁，改为给 `@neuroloom/core` 配置别名，或调整 `server.watch`。
3. **导入后原始变量名丢失。** fx 只保留数据流，`z`、`h` 等局部变量名会变成 `x`、`dec1`；输出仍然正确，但可读性下降。按 PRD 11.3–11.4，M2 用 LibCST 静态解析后按限定名合并，即可找回。

## 已知限制（按计划留给后续里程碑）

- 只支持单个扁平模块；用户自定义子模块会被展平，带 `.` 的子模块名会改名并改变 `state_dict` 键名（导入时给出警告）。层级保留在 M2。
- 不支持的算子直接报错并列出清单；降级为代码块节点在 M2。
- 模型必须能用默认参数构造；依赖输入值的控制流会被 fx 拒绝并提示。
- 没有撤销 / 重做、`.loom` 保存 / 打开、快捷键；代码预览是自带的轻量高亮，不是 Monaco。都在 M1。

## 与 PRD 的偏差

| 项 | PRD | M0 实际 | 原因 |
| --- | --- | --- | --- |
| 包管理 | pnpm workspaces | npm workspaces | 本机未装 pnpm，npm 零额外配置；M1 可无痛切换 |
| 桌面壳 | Electron | Vite 开发服务器 + Web 页面 | M0 只验证核心链路；`PythonBackend` 类已按 Electron 主进程可复用的方式写好 |
| 界面语言 | 中英双语 | 仅中文 | i18n 在 M1 接入 i18next |

## 对 M1 的建议

1. 先接 Electron 壳，把 `apps/studio/dev/pythonBridge.ts` 中的 `PythonBackend` 移到主进程，并加上环境检测与一键创建（PRD F10.1）。
2. 再做撤销 / 重做（Immer 补丁）与 `.loom` 读写；这两项决定 store 的最终形态，越早越好。
3. 扩充节点库时，每个 NodeSpec 都要配一条往返用例：本次的测试框架已经可以直接复用。
4. 往返测试依赖本机 torch，CI 需要缓存 CPU 版 torch 的 wheel。
