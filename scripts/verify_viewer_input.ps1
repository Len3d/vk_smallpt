# Drives vk_viewer with synthetic mouse and keyboard input to prove that the
# viewport really responds to dragging, and that R restores the default camera.
# Writes renders/input_1_default.png ... input_4_reset_denoised.png.
param(
  [string]$Exe = "$PSScriptRoot\..\build\Release\vk_viewer.exe",
  [int]$Width = 1000,
  [int]$Height = 700
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public struct RECT { public int Left, Top, Right, Bottom; }
public struct POINT { public int X, Y; }
public static class Native {
  [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, int dx, int dy, uint d, uint e);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr FindWindowW(string cls, string win);
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr l);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  delegate bool EnumProc(IntPtr h, IntPtr l);
  // Picks the visible top-level window that belongs to the viewer process; the
  // console the app also owns is ignored because it is minimised (not visible).
  public static IntPtr FindByPid(int pid) {
    IntPtr res = IntPtr.Zero;
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p == (uint)pid && IsWindowVisible(h)) { res = h; return false; }
      return true;
    }, IntPtr.Zero);
    return res;
  }
  [DllImport("user32.dll")] public static extern bool PostMessageW(IntPtr h, uint m, IntPtr w, IntPtr l);
}
"@

# Without this the coordinates below would be DPI-virtualised and the screen
# capture would be cropped, which made the first version of this script useless.
[Native]::SetProcessDPIAware() | Out-Null

$outDir = Join-Path $PSScriptRoot '..\renders'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$outDir = (Resolve-Path $outDir).Path

function Save-Window([IntPtr]$hwnd, [string]$path) {
  $r = New-Object RECT
  [Native]::GetWindowRect($hwnd, [ref]$r) | Out-Null
  $w = $r.Right - $r.Left; $h = $r.Bottom - $r.Top
  $bmp = New-Object System.Drawing.Bitmap $w, $h
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($r.Left, $r.Top, 0, 0, $bmp.Size)
  $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
  $g.Dispose(); $bmp.Dispose()
  Write-Host "captured $path ($w x $h)"
}

function Client-ToScreen([IntPtr]$hwnd, [int]$x, [int]$y) {
  $p = New-Object POINT; $p.X = $x; $p.Y = $y
  [Native]::ClientToScreen($hwnd, [ref]$p) | Out-Null
  return $p
}

$proc = Start-Process -FilePath $Exe -ArgumentList "--width", $Width, "--height", $Height -PassThru
try {
  Start-Sleep -Seconds 5
  # MainWindowHandle would be the (minimised) console window, so look the Vulkan
  # window up by its registered class name instead.
  $hwnd = [Native]::FindByPid($proc.Id)
  if ($hwnd -eq [IntPtr]::Zero) { throw 'viewer window not found' }
  [Native]::SetForegroundWindow($hwnd) | Out-Null
  Start-Sleep -Milliseconds 1500
  Save-Window $hwnd (Join-Path $outDir 'input_1_default.png')

  # left drag well inside the viewport (bottom right of the ImGui panel): orbit
  $p = Client-ToScreen $hwnd 760 500
  [Native]::SetCursorPos($p.X, $p.Y) | Out-Null
  Start-Sleep -Milliseconds 300
  [Native]::mouse_event(0x0002, 0, 0, 0, 0)   # left down
  for ($i = 1; $i -le 14; $i++) {
    [Native]::mouse_event(0x0001, -16, 5, 0, 0)  # relative move while held
    Start-Sleep -Milliseconds 40
  }
  [Native]::mouse_event(0x0004, 0, 0, 0, 0)   # left up
  Start-Sleep -Seconds 2
  Save-Window $hwnd (Join-Path $outDir 'input_2_orbited.png')

  # wheel: zoom, cursor kept far away from the ImGui panel so the slider is not hit
  $p = Client-ToScreen $hwnd 820 600
  [Native]::SetCursorPos($p.X, $p.Y) | Out-Null
  Start-Sleep -Milliseconds 300
  [Native]::mouse_event(0x0800, 0, 0, 120, 0)
  Start-Sleep -Milliseconds 400
  [Native]::mouse_event(0x0800, 0, 0, 120, 0)
  Start-Sleep -Seconds 2
  Save-Window $hwnd (Join-Path $outDir 'input_3_zoomed.png')

  # R resets the camera, D runs the denoiser on whatever has accumulated
  [Native]::PostMessageW($hwnd, 0x0100, [IntPtr]0x52, [IntPtr]0) | Out-Null   # 'R'
  Start-Sleep -Milliseconds 1200
  [Native]::PostMessageW($hwnd, 0x0100, [IntPtr]0x44, [IntPtr]0) | Out-Null   # 'D'
  Start-Sleep -Seconds 3
  Save-Window $hwnd (Join-Path $outDir 'input_4_reset_denoised.png')

  # S writes a PNG through the same code path as the Save PNG button
  [Native]::PostMessageW($hwnd, 0x0100, [IntPtr]0x53, [IntPtr]0) | Out-Null   # 'S'
  Start-Sleep -Seconds 2
  Save-Window $hwnd (Join-Path $outDir 'input_5_saved.png')
  $saved = Get-ChildItem (Join-Path $outDir 'viewer_*.png') | Sort-Object LastWriteTime |
           Select-Object -Last 1
  if ($saved) { Write-Host "keyboard save produced $($saved.Name) ($($saved.Length) bytes)" }
  else { Write-Host 'keyboard save produced no file' }
}
finally {
  if ($proc -and -not $proc.HasExited) {
    $viewer = [Native]::FindByPid($proc.Id)
    if ($viewer -ne [IntPtr]::Zero) {
      [Native]::PostMessageW($viewer, 0x0010, [IntPtr]0, [IntPtr]0) | Out-Null   # WM_CLOSE
      Start-Sleep -Seconds 2
    }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
  }
  Write-Host 'viewer closed'
}
