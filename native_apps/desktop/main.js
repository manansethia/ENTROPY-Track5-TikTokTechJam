// AetherForensics — Native Desktop Application Runtime (macOS, Windows, Linux)
const { app, BrowserWindow, dialog, ipcMain, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

// Enable Full Native Hardware Acceleration (Discrete/Integrated GPU, WebGL2, DirectML, Metal)
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-zero-copy');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-webgl2-compute-context');

let mainWindow;
let localBackendProcess = null;

function startEmbeddedLocalEngine() {
  const pythonPath = process.env.AETHER_PYTHON || 'python3';
  const serverScript = path.join(__dirname, '..', '..', 'app', 'server.py');
  
  try {
    localBackendProcess = spawn(pythonPath, ['-m', 'uvicorn', 'app.server:app', '--host', '127.0.0.1', '--port', '8000'], {
      cwd: path.join(__dirname, '..', '..'),
      env: { ...process.env, OMP_NUM_THREADS: String(require('os').cpus().length) }
    });
    console.log('[Native Desktop] Embedded local inference engine started on 127.0.0.1:8000');
  } catch (err) {
    console.log('[Native Desktop] Note: Using existing local backend service:', err.message);
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 880,
    minWidth: 900,
    minHeight: 650,
    title: 'AetherForensics — TikTok AIGC Forensic Studio (Native Desktop Edition)',
    backgroundColor: '#010101',
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    vibrancy: 'ultra-dark',
    visualEffectState: 'active',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    }
  });

  const staticIndexPath = path.join(__dirname, '..', '..', 'app', 'static', 'index.html');
  mainWindow.loadFile(staticIndexPath);

  // Native Menu for macOS / Windows / Linux
  const template = [
    {
      label: 'AetherForensics',
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Image / Video for Forensic Audit...',
          accelerator: 'CmdOrCtrl+O',
          click: async () => {
            const result = await dialog.showOpenDialog(mainWindow, {
              properties: ['openFile'],
              filters: [
                { name: 'Media Files', extensions: ['jpg', 'png', 'webp', 'mp4', 'mov', 'avi'] }
              ]
            });
            if (!result.canceled && result.filePaths.length > 0) {
              mainWindow.webContents.send('file-opened', result.filePaths[0]);
            }
          }
        },
        { type: 'separator' },
        { role: 'close' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  startEmbeddedLocalEngine();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('will-quit', () => {
  if (localBackendProcess) {
    localBackendProcess.kill();
  }
});


app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
