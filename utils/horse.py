import vlc


class horse:
    def __init__(self ):
        self.switch = "idle"
        self.instance = self.init_instance()
        self.save = False

        self.max_tap = 12
        self.curr_tap = 0



    def init_instance(self):
        vlc_inst = vlc.Instance('--no-video-title-show', '--fullscreen','--video-on-top', '--mouse-hide-timeout=0')
        #create media_player
        vlc_inst = vlc.MediaPlayer(vlc_inst)
        vlc_inst.set_fullscreen(True)
        return vlc_inst

    def queue(self):
        if self.instance.is_playing() == 0:      
            match self.switch:
                
                case "idle":
                    media = vlc.Media("datensammeln/looking_around.mp4")
                    print("idleing")
                case "start_tap":
                    media = vlc.Media("datensammeln/tap_loop_start0001-0059.mp4")
                    self.switch = "tapping"
                    self.curr_tap = 1
                case "tapping":
                    print(f"**TAP**  {self.curr_tap}")
                    self.curr_tap += 1
                    media = vlc.Media("datensammeln/tap_loop_start0060-0088.mp4")
                    if self.curr_tap == self.max_tap:
                        self.switch = "end_tap"
                case "end_tap":
                    media = vlc.Media("datensammeln/tap_loop_start0118-0139.mp4")
                    self.switch = "announce_end"
                case "announce_end":
                    media = vlc.Media("datensammeln/tap_loop_start0900-1050.mp4")
                    self.switch = "reset_idle" 
                case "reset_idle":
                    media = media = vlc.Media("datensammeln/looking_around.mp4")
                    self.switch = "idle"
            self.instance.set_media(media)
            self.instance.play()
