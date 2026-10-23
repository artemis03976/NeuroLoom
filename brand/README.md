# NeuroLoom 品牌规范

<img src="assets/neuroloom-lockup-zh-light.svg" height="56" alt="NeuroLoom 织络">

## 概念

**经线是结构，纬线是数据。** 标志是一个被“织”出来的 N：紫色经线构成 N，代表网络结构；青色纬线从两根竖线上方、从斜线下方穿过，代表流过网络的张量，末端圆点指向输出。这与编辑器里“节点是结构、边是流动的张量”一一对应。

## 资源

所有文件都由 `scripts/build.mjs` 从 `palette.json` 生成，改颜色只改 `palette.json`，再运行 `npm run brand`。不要手改生成的文件。

| 文件 | 用途 |
| --- | --- |
| `assets/neuroloom-mark.svg` | 主标志（深色圆角底），应用图标、favicon |
| `assets/neuroloom-mark-light.svg` / `-dark.svg` | 无底版，分别用于浅色 / 深色背景 |
| `assets/neuroloom-mark-mono.svg` | 单色版，颜色取 `currentColor` |
| `assets/neuroloom-lockup-{light,dark}.svg` | 标志 + NeuroLoom 字标 |
| `assets/neuroloom-lockup-zh-{light,dark}.svg` | 标志 + NeuroLoom + 织络 |
| `assets/icons/icon-{16…1024}.png`、`assets/favicon.ico` | 位图图标（Electron 打包在 M1 使用） |
| `tokens.css` | 设计 token（CSS 变量，含浅色 / 深色语义色），编辑器直接引用 |
| `preview.html` | 可视化预览全部资源与色板 |

字标用 Space Grotesk Bold、中文用 Noto Sans SC Bold 转为路径，文件不依赖任何字体（两者均为 OFL 许可）。

## 使用规则

- 标志四周留白不少于标志宽度的 1/4；最小显示尺寸 16 px。
- 不拉伸、不旋转、不改动经纬线的穿插关系，不给标志加阴影或描边。
- 浅色背景用 `-light` 无底版或带底主标志；深色背景用 `-dark` 无底版或带底主标志。

## 色板

### 品牌色

| 名称 | 色值 | 用途 |
| --- | --- | --- |
| Ink 墨 | `#11152A` | 标志底色、深色界面最深层 |
| Warp 经线紫 600 / 400 | `#5A48F0` / `#8B7DFF` | 主色：按钮、选中、链接（浅色 / 深色主题） |
| Weft 纬线青 600 / 400 | `#0AA3B5` / `#3FE0F0` | 数据流：边、流动粒子（浅色 / 深色主题） |

完整色阶（300–700）见 `palette.json`。

### 节点类别色

同一个色值在浅色和深色主题下都使用，用于节点顶边、图标底色和小地图。

| 类别 | 色值 | 类别 | 色值 |
| --- | --- | --- | --- |
| 输入输出 io | `#8C95B0` | 循环 recurrent | `#EE5A92` |
| 线性 linear | `#6576FF` | 注意力 attention | `#B867F5` |
| 卷积 conv | `#3D8BFF` | 张量操作 tensor | `#4CBF6B` |
| 池化 pool | `#31B8E0` | 运算 math | `#A6C93A` |
| 归一化 norm | `#20B5A0` | 模块 module | `#8B7DFF` |
| 激活 activation | `#FF8A3D` | 代码块 code | `#737B90` |
| Dropout | `#A3AABF` | | |

模块节点用经线紫，与“经线 = 结构”的概念一致。

### 状态色

| 状态 | 色值 |
| --- | --- |
| 错误 | `#E5484D` |
| 警告 | `#F5A524` |
| 成功 | `#2FB874` |
| 信息 | `#3D8BFF` |

## 字体

| 用途 | 字体栈 |
| --- | --- |
| 界面 | Segoe UI Variable, PingFang SC, Microsoft YaHei, system-ui |
| 代码、变量名、形状 | JetBrains Mono, Cascadia Code, Consolas, monospace |
| 字标 / 标题展示 | Space Grotesk（仅用于品牌资源，界面不加载） |
