# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules


scipy_binaries = collect_dynamic_libs('scipy')
scipy_hiddenimports = collect_submodules('scipy.signal') + collect_submodules('scipy.linalg') + ['scipy._cyutility']


a = Analysis(
    ['ecg_app/main.py'],
    pathex=['.'],
    binaries=scipy_binaries,
    datas=[('spinner.gif', '.'), ('ecg.ico', '.')],
    hiddenimports=scipy_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ECGViewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['ecg.ico'],
)
