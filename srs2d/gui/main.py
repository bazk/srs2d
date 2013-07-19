# -*- coding: utf-8 -*-
#
# This file is part of srs2d.
#
# srs2d is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# trooper-simulator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with srs2d. If not, see <http://www.gnu.org/licenses/>.

"""
Create the main GTK window.
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "26 Jun 2013"

import os
import gtk
import gobject
import threading
import logging
import scene
from .. import physics

__log__ = logging.getLogger(__name__)

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
        self.set_role("srs2d-viewer")

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
        area_event_box.add_events(gtk.gdk.MOTION_NOTIFY | gtk.gdk.BUTTON_PRESS | gtk.gdk.SCROLL_MASK)
        area_event_box.connect('button-press-event', self.on_mouse_down)
        area_event_box.connect('button-release-event', self.on_mouse_up)
        area_event_box.connect('motion-notify-event', self.on_mouse_move)
        area_event_box.connect("scroll-event", self.on_mouse_scroll)
        area = gtk.DrawingArea()
        area.set_app_paintable(True)
        area.set_size_request(self.scene_resolution[0], self.scene_resolution[1])
        area_event_box.add(area)
        hbox.pack_start(area_event_box, expand=True, fill=False)

        attr_view = ObjectAttributesTreeView()
        hbox.pack_end(attr_view.get_view(), expand=False, fill=True)

        area.realize()

        # Force SDL to write on our drawing area
        os.putenv('SDL_WINDOWID', str(area.window.xid))

        # We need to flush the XLib event loop otherwise we can't
        # access the XWindow which set_mode() requires
        gtk.gdk.flush()

        self.scene = scene.Scene(resolution=self.scene_resolution)
        self.scene.selection_change_callback = attr_view.on_selection_change
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

    def on_mouse_scroll(self, widget, event):
        if self.scene is not None:
            self.scene.on_mouse_scroll(event)

class ObjectAttributesTreeView():
    def __init__(self):
        self.store = gtk.TreeStore(str, object)

        self.view = gtk.TreeView(self.store)
        self.view.set_size_request(230, 120)

        self.renderer = {}
        self.column = {}

        self.renderer0 = gtk.CellRendererText()
        self.column0 = gtk.TreeViewColumn('Key', self.renderer0, text=0)
        self.view.append_column(self.column0)

        self.renderer1 = gtk.CellRendererText()
        self.renderer1.connect('edited', self.cell_edited, self.store)
        self.column1 = gtk.TreeViewColumn('Value', self.renderer1)
        self.column1.set_cell_data_func(self.renderer1, self.cell_data)
        self.view.append_column(self.column1)

    def get_view(self):
        return self.view

    def on_selection_change(self, obj):
        self.store.clear()

        # if obj is not None:
        #     for attr in obj.attributes:
        #         # value = attr.getter()

        #         # if isinstance(value, tuple):
        #         #     root = self.store.append(None, (attr.key, value, attr))
        #         #     i = 0
        #         #     for part in value:
        #         #         self.store.append(root, (i, value, attr))
        #         #         i += 1

        #         # elif isinstance(value, physics.Vector):
        #         #     x, y = (value.x, value.y)
        #         #     root = self.store.append(None, (attr.key, value, attr))
        #         #     self.store.append(root, ('x', value, attr))
        #         #     self.store.append(root, ('y', value, attr))

        #         # else:
        #         self.store.append(None, (attr.key, attr))

    def cell_data(self, column, cell, model, iter):
        key = model.get_value(iter, 0)
        attr = model.get_value(iter, 1)
        cell.set_property('editable', not attr.read_only)
        cell.set_property('text', attr.getter())

    def cell_edited(self, cell, path, new_text, model):
        row = model[path]
        key = row[0]
        attr = row[1]
        attr.setter(eval(new_text))
        cell.set_property('text', attr.getter())