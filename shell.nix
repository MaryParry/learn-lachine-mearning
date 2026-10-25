{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "python-ml-env";
  targetPkgs = pkgs: (with pkgs; [
    uv
    python3
    gcc
    stdenv.cc.cc.lib
    zlib
    glibc

    # GUI and Qt dependencies for matplotlib & pyqt6
    libGL
    glib
    zstd
    fontconfig
    freetype
    dbus
    libxkbcommon
    wayland
    libx11
    libxext
    libxrender
    libxcomposite
    libxdamage
    libxfixes
    libxi
    libxrandr
    libxtst
    libxcb
    libxcb-cursor
    libxcb-image
    libxcb-keysyms
    libxcb-render-util
    libxcb-wm
  ]);
  runScript = "fish";
}).env
