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

export function buildMenu(recent: string[], locale: 'zh' | 'en'): void {
  const t = (zh: string, en: string) => (locale === 'en' ? en : zh);
  const isMac = process.platform === 'darwin';
  const template: MenuItemConstructorOptions[] = [
    ...(isMac ? [{ role: 'appMenu' } as MenuItemConstructorOptions] : []),
    {
      label: t('文件', 'File'),
      submenu: [
        item(t('新建', 'New'), 'new', 'CmdOrCtrl+N'),
        item(t('打开…', 'Open…'), 'open', 'CmdOrCtrl+O'),
        {
          label: t('打开最近', 'Open Recent'),
          submenu: recent.length
            ? recent.map((path) => ({ label: path, click: send(`openRecent:${path}`) }))
            : [{ label: t('（无）', '(none)'), enabled: false }],
        },
        { type: 'separator' },
        item(t('保存', 'Save'), 'save', 'CmdOrCtrl+S'),
        item(t('另存为…', 'Save As…'), 'saveAs', 'CmdOrCtrl+Shift+S'),
        { type: 'separator' },
        item(t('从代码导入…', 'Import from Code…'), 'import', 'CmdOrCtrl+I'),
        item(t('导出 .py…', 'Export .py…'), 'exportPy', 'CmdOrCtrl+E'),
        item(t('导出选项…', 'Export Options…'), 'exportOptions'),
        item(t('导出图片…', 'Export Image…'), 'exportImage'),
        { type: 'separator' },
        item(t('Python 环境…', 'Python Environment…'), 'environment'),
        { type: 'separator' },
        isMac ? { role: 'close', label: t('关闭窗口', 'Close Window') } : { role: 'quit', label: t('退出', 'Quit') },
      ],
    },
    {
      label: t('编辑', 'Edit'),
      submenu: [
        item(t('撤销', 'Undo'), 'undo', 'CmdOrCtrl+Z'),
        item(t('重做', 'Redo'), 'redo', 'CmdOrCtrl+Shift+Z'),
        { type: 'separator' },
        { role: 'cut', label: t('剪切', 'Cut') },
        { role: 'copy', label: t('复制', 'Copy') },
        { role: 'paste', label: t('粘贴', 'Paste') },
        { role: 'selectAll', label: t('全选', 'Select All') },
      ],
    },
    {
      label: t('模块', 'Module'),
      submenu: [
        item(t('打包为模块…', 'Pack as Module…'), 'packModule', 'CmdOrCtrl+G'),
        item(t('解包', 'Unpack'), 'unpackModule', 'CmdOrCtrl+Shift+G'),
        { type: 'separator' },
        item(t('新建模块…', 'New Module…'), 'newModule'),
        item(t('返回上一级', 'Up One Level'), 'goUp', 'Alt+Left'),
      ],
    },
    {
      label: t('视图', 'View'),
      submenu: [
        item(t('自动布局', 'Auto Layout'), 'autoLayout', 'CmdOrCtrl+L'),
        item(t('网格吸附', 'Snap to Grid'), 'toggleSnap'),
        item(t('前向回放', 'Forward Replay'), 'replay'),
        item(t('收起/展开代码与分析面板', 'Toggle Code & Analysis Panel'), 'toggleBottomPanel', 'CmdOrCtrl+J'),
        { type: 'separator' },
        {
          label: t('语言', 'Language'),
          submenu: [
            { label: '中文', type: 'radio', checked: locale === 'zh', click: send('locale:zh') },
            { label: 'English', type: 'radio', checked: locale === 'en', click: send('locale:en') },
          ],
        },
        { type: 'separator' },
        { role: 'resetZoom', label: t('实际大小', 'Actual Size') },
        { role: 'zoomIn', label: t('放大', 'Zoom In') },
        { role: 'zoomOut', label: t('缩小', 'Zoom Out') },
        { type: 'separator' },
        { role: 'togglefullscreen', label: t('全屏', 'Full Screen') },
        ...(app.isPackaged ? [] : ([{ type: 'separator' }, { role: 'reload' }, { role: 'toggleDevTools' }] as MenuItemConstructorOptions[])),
      ],
    },
    {
      label: t('帮助', 'Help'),
      submenu: [{ label: t('PyTorch 文档', 'PyTorch Documentation'), click: () => void shell.openExternal('https://docs.pytorch.org/docs/stable/') }],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}
