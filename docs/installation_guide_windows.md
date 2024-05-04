# How to install Cremage

Currently, only systems with NVIDIA GPU and CUDA are supported.

# Overview
1. Install required tools
2. Build GTK3/PyGObject
3. Copy Cremage files from github
4. Set up conda environment
5. Install Pytorch
6. Install GTK
7. Install other python packages
8. Copy model files

# Steps
## Install required tools

   Installing Cremage on your system requires following tools:
   * Python
   * Anaconda
   * Chocolatey
   * Git
   * MSYS2
   * Microsoft VisualStudio

Chocolatey is an application that is used for installing third party packages.
Git is used to download Cremage code fom Github. MSYS2 is required to build Gtk. Anaconda is used for installing Python packages. Microsoft VisualStudio is required to build Gtk and PyGObject which is a UI framework that Cremage uses to manage UI.
Fortunately, Chocolatey can handle installation for some of these packages without you going through tedeous installation process.

### Installing Python
1. Go to https://www.python.org/downloads/release/python-31011/
2. Download Windows Installer 64bit for Python 3.10.11
3. Once the installer is downloaded, run it from your Downloads folder to install.

### Installing Anaconda
1. Go to https://www.anaconda.com/download
1. Select 64-Bit Graphical Installer for Python 3.11 and download. Note that we will be using Python 3.10, but you only see Python 3.11 on this page. It's OK to use Python 3.11 for this part.
1. Run the downloaded installer from the Downloads folder to install.
1. Anaconda should be installed to:
   c:\Users\<Your user name>\anaconda3\
   We need to add Anaconda to PATH, so select Environronment Variable for your account from Control Panel. You should see two lists on the screen. Add the following to PATH in the list on top and save:
   c:\Users\<Your user name>\anaconda3\Scripts

### Installing Chocolatey
Note that installing Chocolatey, packages using Chocolatey, building GTK & PyGObject are taken from https://github.com/wingtk/gvsbuild, so credit goes to the author of that page.

1. Right-click PowerShell icon and select a menu item to run it as an adminitrator.
2. Type the following to download and install the application:
   ```
    Set-ExecutionPolicy Bypass -Scope Process -Force; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```
   Refer to https://chocolatey.org/install for more information.

   To allow unsigned local script execution, type:
   ```
   Set-ExecutionPolicy RemoteSigned
   ```

3. Once above command is completed, do not close the terminal and type the following to install git and msys2
   ```
    choco install git
    choco install msys2
    choco install visualstudio2022-workload-vctools
   ```

## Building GTK/PyGObject
Close the PowerShell window if it's still open from the previous step.
Now open the new PowerShell as a regular (non-administrator) user just by clicking the PowerShell icon.
Type:
```
py -3.10 -m pip install --user pipx
py -3.10 -m pipx ensurepath
```

Close powershell and reopen powershell again. Then type:
```
pipx install gvsbuild
```

Then type:
```
$env:LIB = "C:\gtk-build\gtk\x64\release\lib;" + $env:LIB
$env:INCLUDE = "C:\gtk-build\gtk\x64\release\include;C:\gtk-build\gtk\x64\release\include\cairo;C:\gtk-build\gtk\x64\release\include\glib-2.0;C:\gtk-build\gtk\x64\release\include\gobject-introspection-1.0;C:\gtk-build\gtk\x64\release\lib\glib-2.0\include;" + $env:INCLUDE
gvsbuild build --enable-gi --py-wheel gtk3 pygobject
```

You should see the output like below:
```
Project(s) built:
    ninja                     (0.006 s)
    meson                     (1.240 s)
    pkgconf                   (2.660 s)
    cmake                     (1.449 s)
    win-iconv                 (2.773 s)
    gettext                   (24.273 s)
    libffi                    (1.728 s)
    zlib                      (1.518 s)
    pcre2                     (5.789 s)
    glib-base                 (20.284 s)
    msys2                     (0.001 s)
    gobject-introspection     (15.442 s)
    glib                      (25.800 s)
    atk                       (4.501 s)
    nasm                      (0.008 s)
    libjpeg-turbo             (7.964 s)
    libtiff-4                 (10.197 s)
    libpng                    (1.625 s)
    gdk-pixbuf                (9.432 s)
    freetype                  (4.212 s)
    pixman                    (3.974 s)
    cairo                     (17.940 s)
    harfbuzz                  (27.872 s)
    fribidi                   (2.422 s)
    pango                     (6.311 s)
    libepoxy                  (4.498 s)
    gtk3                      (44.627 s)
    pycairo                   (9.319 s)
    pygobject                 (12.127 s)
```

### Add GTK to Your Environmental Variables
Add the following to your PATH:
```
C:\gtk-build\gtk\x64\release\bin
```
Refer to the Anaconda section earlier about how to update PATH.

## Copy Cremage files from github

   Open the terminal and go to a directory that you want to install Cremage.
   Type:

```
git clone https://github.com/HowToSD/cremage.git
```

## Set up conda environment

Run the following. Note that if your system does not have conda, you need to install it first.

```
conda create -n cremage python=3.10
conda activate cremage
```

## Install Pytorch
Type:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
For more information, refer to the installation section at https://pytorch.org/.

## Install GTK
This step requires the GTK/PyGObject have been successfully built in an earlier step.
Type:
```
pip install --force-reinstall (Resolve-Path C:\gtk-build\build\x64\release\pygobject\dist\PyGObject*.whl)
pip install --force-reinstall (Resolve-Path C:\gtk-build\build\x64\release\pycairo\dist\pycairo*.whl)
```

## Install other python packages
Type:
```
pip install -r requirements_win.txt
```

## Copy model files
See [Setting up model files](setting_up_model_files.md).

