import os
import gtk
import gobject
import threading
import scene

gobject.threads_init()

def get_resource_path(rel_path):
    dir_of_py_file = os.path.dirname(__file__)
    rel_path_to_resource = os.path.join(dir_of_py_file, rel_path)
    abs_path_to_resource = os.path.abspath(rel_path_to_resource)
    return abs_path_to_resource

class Main(gtk.Window):
    scene_resolution = (800, 600)

    def __init__(self):
        super(Main, self).__init__()
        self.set_name("SRS2D Viewer")

        self.connect("destroy", self.exit)

        self.icon_play = gtk.image_new_from_file(get_resource_path("icons/play.xpm"))
        self.icon_play.show()
        self.icon_pause = gtk.image_new_from_file(get_resource_path("icons/pause.xpm"))
        self.icon_pause.show()
        self.icon_loop = gtk.image_new_from_file(get_resource_path("icons/loop.xpm"))
        self.icon_loop.show()

        vbox = gtk.VBox()
        self.add(vbox)

        toolbar = gtk.Toolbar()
        toolbar.set_orientation(gtk.ORIENTATION_HORIZONTAL)
        toolbar.set_style(gtk.TOOLBAR_ICONS)
        toolbar.set_border_width(5)

        self.step_button = gtk.ToolButton(label='Step')
        self.step_button.set_icon_widget(self.icon_loop)
        self.step_button.connect('clicked', self.on_step_clicked)
        toolbar.insert(self.step_button, 0)

        self.start_stop_button = gtk.ToolButton(label='Start')
        self.start_stop_button.set_icon_widget(self.icon_play)
        self.start_stop_button.connect('clicked', self.on_start_stop_clicked)
        toolbar.insert(self.start_stop_button, 1)

        vbox.pack_start(toolbar, expand=False, fill=False)

        hbox = gtk.HBox()
        vbox.pack_start(hbox, expand=True, fill=True)

        area_event_box = gtk.EventBox()
        area_event_box.add_events(gtk.gdk.MOTION_NOTIFY | gtk.gdk.BUTTON_PRESS)
        area_event_box.connect('button-press-event', self.on_mouse_down)
        area_event_box.connect('button-release-event', self.on_mouse_up)
        area_event_box.connect('motion-notify-event', self.on_mouse_move)
        area = gtk.DrawingArea()
        area.set_app_paintable(True)
        area.set_size_request(self.scene_resolution[0], self.scene_resolution[1])
        area_event_box.add(area)
        hbox.pack_start(area_event_box, expand=True, fill=False)

        area.realize()

        # Force SDL to write on our drawing area
        os.putenv('SDL_WINDOWID', str(area.window.xid))

        # We need to flush the XLib event loop otherwise we can't
        # access the XWindow which set_mode() requires
        gtk.gdk.flush()

        self.scene = scene.Scene(resolution=self.scene_resolution)
        gobject.idle_add(self.draw_scene)

        self.show_all()

        gtk.main()

    def exit(self, widget, data=None):
        gtk.main_quit()

    def draw_scene(self):
        try:
            self.scene.draw()
            gobject.idle_add(self.draw_scene)
        except KeyboardInterrupt:
            gtk.main_quit()

    def on_step_clicked(self, button):
        if self.scene is None:
            return

        if self.scene.is_running():
            self.scene.stop()
            self.start_stop_button.set_label('Start')
            self.start_stop_button.set_icon_widget(self.icon_play)

        self.scene.step()

    def on_start_stop_clicked(self, button):
        if self.scene is None:
            return

        if self.scene.is_running():
            self.scene.stop()
            self.start_stop_button.set_label('Start')
            self.start_stop_button.set_icon_widget(self.icon_play)
        else:
            self.scene.start()
            self.start_stop_button.set_label('Stop')
            self.start_stop_button.set_icon_widget(self.icon_pause)

    def on_mouse_down(self, widget, event):
        if self.scene is not None:
            self.scene.on_mouse_down(event)

    def on_mouse_up(self, widget, event):
        if self.scene is not None:
            self.scene.on_mouse_up(event)

    def on_mouse_move(self, widget, event):
        if self.scene is not None:
            self.scene.on_mouse_move(event)