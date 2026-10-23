import { app, BrowserWindow, Menu, shell, type MenuItemConstructorOptions } from 'electron';
import { CHANNELS, type MenuCommand } from '@neuroloom/host-api';

const send = (command: MenuCommand) => () => BrowserWindow.getFocusedWindow()?.webContents.send(CHANNELS.command, command);

/**
 * Shortcuts are handled in the renderer for both the desktop and web builds,
 * so menu accelerators are shown but not registered (registerAccelerator: false).
 */
const item = (label: string, command: MenuCommand, accelerator?: string): MenuItemConstructorOptions => ({
  label,
  accelerator,
  registerAccelerator: false,
  click: send(command),
});

export function buildMenu(recent: string[]): void {
  const isMac = process.platform === 'darwin';
  const template: MenuItemConstructorOptions[] = [
    ...(isMac ? [{ role: 'appMenu' } as MenuItemConstructorOptions] : []),
    {
      label: '文件',
      submenu: [
        item('新建', 'new', 'CmdOrCtrl+N'),
        item('打开…', 'open', 'CmdOrCtrl+O'),
        {
          label: '打开最近',
          submenu: recent.length
            ? recent.map((path) => ({ label: path, click: send(`openRecent:${path}`) }))
            : [{ label: '（无）', enabled: false }],
        },
        { type: 'separator' },
        item('保存', 'save', 'CmdOrCtrl+S'),
        item('另存为…', 'saveAs', 'CmdOrCtrl+Shift+S'),
        { type: 'separator' },
        item('从代码导入…', 'import', 'CmdOrCtrl+I'),
        item('导出 .py…', 'exportPy', 'CmdOrCtrl+E'),
        { type: 'separator' },
        item('Python 环境…', 'environment'),
        { type: 'separator' },
        isMac ? { role: 'close', label: '关闭窗口' } : { role: 'quit', label: '退出' },
      ],
    },
    {
      label: '编辑',
      submenu: [
        item('撤销', 'undo', 'CmdOrCtrl+Z'),
        item('重做', 'redo', 'CmdOrCtrl+Shift+Z'),
        { type: 'separator' },
        { role: 'cut', label: '剪切' },
        { role: 'copy', label: '复制' },
        { role: 'paste', label: '粘贴' },
        { role: 'selectAll', label: '全选' },
      ],
    },
    {
      label: '视图',
      submenu: [
        item('自动布局', 'autoLayout', 'CmdOrCtrl+L'),
        { type: 'separator' },
        { role: 'resetZoom', label: '实际大小' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { type: 'separator' },
        { role: 'togglefullscreen', label: '全屏' },
        ...(app.isPackaged ? [] : ([{ type: 'separator' }, { role: 'reload' }, { role: 'toggleDevTools' }] as MenuItemConstructorOptions[])),
      ],
    },
    {
      label: '帮助',
      submenu: [{ label: 'PyTorch 文档', click: () => void shell.openExternal('https://docs.pytorch.org/docs/stable/') }],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
