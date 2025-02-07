# Undocumented Pyside6 system dependencies
sudo apt install -y \
            libfontconfig1-dev libfreetype6-dev \
            libx11-dev libx11-xcb-dev libxext-dev libxfixes-dev \
            libxi-dev libxrender-dev \
            libxkbcommon-dev libxkbcommon-x11-dev libatspi2.0-dev \
            libopengl0 '^libxcb.*-dev'
            libgl1 \
            libegl1 \
            libdbus-glib-1-2

Then set: export QT_QPA_PLATFORM="xcb"
