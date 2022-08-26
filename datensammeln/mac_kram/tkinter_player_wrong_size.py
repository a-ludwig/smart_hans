import os
import platform
import sys
import tkinter
from ctypes import c_void_p, cdll
from threading import Thread

import vlc

system = platform.system()

if system == "Darwin":
    # find the accurate Tk lib for Mac
    libtk = "libtk%s.dylib" % (tkinter.TkVersion,)
    if "TK_LIBRARY_PATH" in os.environ:
        libtk = os.path.join(os.environ["TK_LIBRARY_PATH"], libtk)
    else:
        prefix = getattr(sys, "base_prefix", sys.prefix)
        libtk = os.path.join(prefix, "lib", libtk)
    dylib = cdll.LoadLibrary(libtk)
    _GetNSView = dylib.TkMacOSXGetRootControl
    _GetNSView.restype = c_void_p
    _GetNSView.argtypes = (c_void_p,)
    del dylib


class Window(tkinter.Tk):
    def register(self, player):
        self.attributes("-fullscreen", True)
        id = self.winfo_id()
        print(id)

        if system == "Darwin":
            player.set_nsobject(_GetNSView(id))
        elif system == "Linux":
            player.set_xwindow(id)
        elif system == "Windows":
            player.set_hwnd(id)


def play(instance, player, path):
    media = instance.media_new_path(path)
    player.set_media(media)
    player.play()


if __name__ == "__main__":
    instance = vlc.Instance()
    player = instance.media_player_new()
    window = Window()
    window.register(player)
    thread = Thread(target=play, args=(instance, player, "Static_664_1080.mp4"))
    thread.start()
    window.mainloop()