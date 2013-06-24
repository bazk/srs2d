import gtk
import gobject
import threading
import scene

class Main(gtk.Window):
    def __init__(self):
        super(Main, self).__init__()
        self.set_name("SRS2D Viewer")

        self.connect("destroy", self.exit)

        vbox = gtk.VBox()
        self.add(vbox)

        self.step_button = gtk.Button(label='Step')
        vbox.pack_start(self.step_button)
        self.start_stop_button = gtk.Button(label='Start')
        vbox.pack_start(self.start_stop_button)

        self.step_button.connect('clicked', self.on_step_clicked)
        self.start_stop_button.connect('clicked', self.on_start_stop_clicked)

        self.scene = None

        self.scene = scene.Scene()
        self.scene.exit_callback = self.scene_exit
        threading.Thread(target=self.scene.run).start()

        self.show_all()

        try:
            gtk.main()
        except KeyboardInterrupt:
            if self.scene is not None:
                self.scene.exit()

    def exit(self, widget, data=None):
        if self.scene is not None:
            self.scene.exit()

        gtk.main_quit()

    def scene_exit(self):
        self.scene = None

    def on_step_clicked(self, button):
        if self.scene is None:
            return

        if self.scene.is_running():
            self.scene.stop()

        self.scene.step()

    def on_start_stop_clicked(self, button):
        if self.scene is None:
            return

        if self.scene.is_running():
            self.scene.stop()
            self.start_stop_button.label = 'Start'
        else:
            self.scene.start()
            self.start_stop_button.label = 'Stop'