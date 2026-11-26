"""Build local release 0.3.0; no download or repository modification."""
from pathlib import Path
import argparse, os, shutil, subprocess, sys, tempfile, zipfile
ROOT=Path(__file__).resolve().parent

def msvc_environment():
    """Use installed VS C++ tools without changing the PowerShell execution policy."""
    if sys.platform != 'win32' or (shutil.which('cl.exe') and shutil.which('link.exe')):
        return
    installer=Path(os.environ.get('ProgramFiles(x86)',r'C:\Program Files (x86)'))/'Microsoft Visual Studio'/'Installer'/'vswhere.exe'
    if not installer.is_file():
        raise SystemExit('MSVC not found: install Visual Studio Build Tools with Desktop development with C++.')
    location=subprocess.check_output([str(installer),'-latest','-products','*','-requires','Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath','-utf8']).decode('utf-8-sig').strip()
    batch=Path(location)/'VC'/'Auxiliary'/'Build'/'vcvars64.bat'
    if not location or not batch.is_file():
        raise SystemExit('MSVC C++ x64 tools not found. Install them with Visual Studio Installer.')
    # cmd /u makes the output of the internal SET command UTF-16LE.
    command=f'""{batch}" >nul && set"'
    cmdline=subprocess.list2cmdline([os.environ.get('COMSPEC','cmd.exe')])+' /d /u /s /c '+command
    output=subprocess.check_output(cmdline).decode('utf-16-le')
    for line in output.splitlines():
        name,sep,value=line.partition('=')
        if sep and name:os.environ[name]=value
    if not shutil.which('cl.exe') or not shutil.which('link.exe'):
        raise SystemExit('MSVC environment could not be initialized. Run vcvars64.bat in Developer Command Prompt.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true',help='Run full installed-wheel tests (requires pytest, Qt/OpenGL and a display)')
    p.add_argument('--out',type=Path,default=ROOT/'dist')
    p.add_argument('--fresh',action='store_true',help='Recompile the local Rust crate even if archived source timestamps are older than the cached build')
    args=p.parse_args()
    if not shutil.which('cargo') or not shutil.which('rustc'):
        raise SystemExit('Install Rust via rustup and reopen the terminal.')
    msvc_environment()
    if args.fresh:
        subprocess.run(['cargo','clean','--release','--offline','-p','unit-cell-gui-native'],cwd=ROOT,check=True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','maturin','build','--release','--offline','--locked','--interpreter',sys.executable,'--out',str(out)]
    if sys.platform.startswith('linux'):cmd+=['--compatibility','linux']
    subprocess.run(cmd,cwd=ROOT,check=True)
    wheels=list(out.glob('unit_cell_gui-0.3.0-*.whl'))
    if len(wheels)!=1:raise RuntimeError('Expected exactly one 0.3.0 wheel in '+str(out))
    wheel=wheels[0]
    if args.verify:
        with tempfile.TemporaryDirectory(prefix='ucg-release-') as name:
            with zipfile.ZipFile(wheel) as z:z.extractall(name)
            env=os.environ.copy();env['PYTHONPATH']=name;env['UNIT_CELL_GUI_SORT_BACKEND']='rust';env['UNIT_CELL_GUI_HATCH_BACKEND']='glsl';env['UNIT_CELL_GUI_CONTACT_BACKEND']='rust';env['UNIT_CELL_GUI_ATOM_BACKEND']='glsl';env['UNIT_CELL_GUI_RENDER_BACKEND']='rust';env['UNIT_CELL_GUI_FRAME_BACKEND']='rust';env['UNIT_CELL_GUI_RESOURCE_BACKEND']='rust'
            env['UNIT_CELL_GUI_WBOIT_BACKEND']='auto'
            env.setdefault('QT_QPA_PLATFORM','windows' if sys.platform=='win32' else 'xcb' if sys.platform.startswith('linux') else 'cocoa')
            subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')],cwd=name,env=env,check=True)
    print('Wheel:',wheel)
if __name__=='__main__':main()
