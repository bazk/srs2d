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
import logging
import ast
import pickle
import scene
from .. import physics
import pyopencl as cl

import pygtk
pygtk.require('2.0')
import gtk
import gobject

__log__ = logging.getLogger(__name__)

gobject.threads_init()



def get_resource_path(rel_path):
    dir_of_py_file = os.path.dirname(__file__)
    rel_path_to_resource = os.path.join(dir_of_py_file, rel_path)
    abs_path_to_resource = os.path.abspath(rel_path_to_resource)
    return abs_path_to_resource

class Main(gtk.Window):
    scene_resolution = (800, 600)

    ui_string = """<ui>
  <menubar name='Menubar'>
    <menu action='FileMenu'>
      <menuitem action='New'/>
      <menuitem action='Open'/>
      <menuitem action='Save'/>
      <separator/>
      <menuitem action='Quit'/>
    </menu>
    <menu action='SimulationMenu'>
      <menuitem action='PlayPause'/>
      <menuitem action='Step'/>
      <menuitem action='IncreaseSpeed'/>
      <menuitem action='DecreaseSpeed'/>
    </menu>
    <menu action='HelpMenu'>
      <menuitem action='About'/>
    </menu>
  </menubar>
  <toolbar name='Toolbar'>
    <toolitem action='New'/>
    <toolitem action='Open'/>
    <toolitem action='Save'/>
    <separator/>
    <toolitem action='PlayPause'/>
    <toolitem action='Step'/>
    <toolitem action='IncreaseSpeed'/>
    <toolitem action='DecreaseSpeed'/>
  </toolbar>
</ui>"""

    def __init__(self):
        super(Main, self).__init__()

        self.simulation = None

        self.set_name("SRS2D Viewer")
        self.set_role("srs2d-viewer")
        self.set_size_request(880, 610)

        self.connect("destroy", self.exit)

        vbox = gtk.VBox()
        self.add(vbox)
        vbox.show()

        ui = gtk.UIManager()
        ui.add_ui_from_string(self.ui_string)

        ag = gtk.ActionGroup('AppActions')
        ag.add_actions([
            ('FileMenu', None, '_File'),
            ('New',      gtk.STOCK_NEW, '_New Simulation', '<control>N', 'Create a new file', self.on_new_clicked),
            ('Open',     gtk.STOCK_OPEN, '_Open', '<control>O', 'Open a file', self.on_open_clicked),
            ('Save',     gtk.STOCK_SAVE, '_Save', '<control>S', 'Save a file', self.on_save_clicked),
            ('Quit',     gtk.STOCK_QUIT, '_Quit', '<control>Q', 'Quit application', None),
            ('SimulationMenu',  None, '_Simulation'),
            ('PlayPause',       None, '_Play/Pause', '<control>P', 'Play/pause the simulation', self.on_playpause_clicked),
            ('Step',            None, 'S_tep', '<control>T', 'Step the simulation', self.on_step_clicked),
            ('IncreaseSpeed',            None, 'Speed +', '<control>plus', 'Increase the speed of simulation', self.on_increase_clicked),
            ('DecreaseSpeed',            None, 'Speed -', '<control>minus', 'Decrease the speed of simulation', self.on_decrease_clicked),
            ('HelpMenu', None, '_Help'),
            ('About',    None, '_About', None, 'About application', None),
        ])
        ui.insert_action_group(ag, 0)

        self.add_accel_group(ui.get_accel_group())

        menubar = ui.get_widget('/Menubar')
        vbox.pack_start(menubar, expand=False)
        menubar.show()

        toolbar = ui.get_widget('/Toolbar')
        vbox.pack_start(toolbar, expand=False)
        toolbar.realize()
        toolbar.show()

        status = gtk.Statusbar()
        vbox.pack_end(status, expand=False)
        status.show()

        self.icon_play = gtk.image_new_from_file(get_resource_path("icons/play.xpm"))
        self.icon_play.show()
        self.icon_pause = gtk.image_new_from_file(get_resource_path("icons/pause.xpm"))
        self.icon_pause.show()
        self.icon_loop = gtk.image_new_from_file(get_resource_path("icons/loop.xpm"))
        self.icon_loop.show()

        self.playpause_button = ui.get_widget('/Toolbar/PlayPause')
        self.playpause_button.set_label('Start')
        self.playpause_button.set_icon_widget(self.icon_play)

        self.step_button = ui.get_widget('/Toolbar/Step')
        self.step_button.set_icon_widget(self.icon_loop)

        hbox = gtk.HBox()
        hbox.show()

        area_event_box = gtk.EventBox()
        area_event_box.add_events(gtk.gdk.MOTION_NOTIFY | gtk.gdk.BUTTON_PRESS | gtk.gdk.SCROLL_MASK)
        area_event_box.connect('button-press-event', self.on_mouse_down)
        area_event_box.connect('button-release-event', self.on_mouse_up)
        area_event_box.connect('motion-notify-event', self.on_mouse_move)
        area_event_box.connect("scroll-event", self.on_mouse_scroll)
        area_event_box.show()

        area = gtk.DrawingArea()
        area.set_app_paintable(True)
        area.set_size_request(self.scene_resolution[0], self.scene_resolution[1])
        area.show()

        vbox.pack_start(hbox, expand=True, fill=True)
        hbox.pack_start(area_event_box, expand=True, fill=False)
        area_event_box.add(area)

        area.realize()

        # Force SDL to write on our drawing area
        os.putenv('SDL_WINDOWID', str(area.window.xid))

        # We need to flush the XLib event loop otherwise we can't
        # access the XWindow which set_mode() requires
        gtk.gdk.flush()

        self.scene = scene.Scene(resolution=self.scene_resolution)
        gobject.timeout_add(1000/30, self.draw_scene)

        self.show()

        gtk.main()

    def exit(self, widget, data=None):
        gtk.main_quit()

    def draw_scene(self):
        try:
            self.scene.draw()
            gobject.timeout_add(1000/30, self.draw_scene)
        except KeyboardInterrupt:
            gtk.main_quit()

    def on_step_clicked(self, button):
        if self.simulation is None:
            return

        if self.simulation.running:
            self.simulation.stop()
            self.playpause_button.set_label('Start')
            self.playpause_button.set_icon_widget(self.icon_play)

        self.simulation.step()

    def on_playpause_clicked(self, button):
        if self.simulation is None:
            return

        if self.simulation.running:
            self.simulation.stop()
            self.playpause_button.set_label('Start')
            self.playpause_button.set_icon_widget(self.icon_play)
        else:
            self.simulation.start()
            self.playpause_button.set_label('Stop')
            self.playpause_button.set_icon_widget(self.icon_pause)

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

    def on_new_clicked(self, button):
        dialog = NewSimulationDialog(self)

        if dialog.run() == gtk.RESPONSE_ACCEPT:
            tb = dialog.params.get_buffer()
            params = ast.literal_eval(tb.get_text(*tb.get_bounds()))
            numbots = int(dialog.numbots.get_text())
            duration = int(dialog.duration.get_text())

            self.simulation = Simulation(self)
            self.simulation.simulate(params, numbots, duration, 1/30.0)

        dialog.destroy()

    def on_increase_clicked(self, button):
        if self.simulation is not None:
            if self.simulation.speed < 8:
                self.simulation.speed += 1

    def on_decrease_clicked(self, button):
        if self.simulation is not None:
            if self.simulation.speed > 1:
                self.simulation.speed -= 1

    def on_open_clicked(self, button):
        chooser = gtk.FileChooserDialog(title='Open simulation replay',
                    action=gtk.FILE_CHOOSER_ACTION_OPEN,
                    buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                             gtk.STOCK_OPEN,gtk.RESPONSE_OK))

        if chooser.run() == gtk.RESPONSE_OK:
            self.simulation = Simulation(self)
            f = open(chooser.get_filename())
            self.simulation.transforms = pickle.load(f)
            f.close()

            self.scene.real_clock, self.scene.transforms = self.simulation.transforms[0]
            self.simulation.current = 0

        chooser.destroy()

    def on_save_clicked(self, button):
        if self.simulation is None:
            return

        chooser = gtk.FileChooserDialog(title='Save simulation replay',
                    action=gtk.FILE_CHOOSER_ACTION_SAVE,
                    buttons=(gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
                             gtk.STOCK_SAVE,gtk.RESPONSE_OK))

        if chooser.run() == gtk.RESPONSE_OK:
            f = open(chooser.get_filename(), 'w')
            pickle.dump(self.simulation.transforms, f)
            f.close()

        chooser.destroy()

class NewSimulationDialog(gtk.Dialog):
    def __init__(self, parent):
        super(NewSimulationDialog, self).__init__("New Simulation", parent, gtk.DIALOG_MODAL,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_REJECT,
             gtk.STOCK_OK, gtk.RESPONSE_ACCEPT))

        self.set_default_size(450, 300)

        box = self.get_content_area()

        vbox = gtk.VBox()
        vbox.show()
        box.add(vbox)

        annframe = gtk.Frame('Neural Network Parameters')
        annframe.show()
        vbox.pack_start(annframe)

        sw = gtk.ScrolledWindow()
        sw.set_policy(gtk.POLICY_AUTOMATIC, gtk.POLICY_AUTOMATIC)
        sw.show()
        annframe.add(sw)

        self.params = gtk.TextView()
        self.params.set_wrap_mode(gtk.WRAP_CHAR)
        self.params.set_left_margin(1)
        self.params.show()
        sw.add(self.params)

        simframe = gtk.Frame('Simulation Parameters')
        simframe.show()
        vbox.pack_start(simframe)

        table = gtk.Table(2, 2)
        table.show()
        simframe.add(table)

        numbots_label = gtk.Label('# of robots: ')
        numbots_label.show()
        table.attach(numbots_label, 0, 1, 0, 1)
        self.numbots = gtk.Entry(max=8)
        self.numbots.set_text("30")
        self.numbots.show()
        table.attach(self.numbots, 1, 2, 0, 1)

        duration_label = gtk.Label('Duration (in seconds): ')
        duration_label.show()
        table.attach(duration_label, 0, 1, 1, 2)
        self.duration = gtk.Entry(max=12)
        self.duration.set_text("600")
        self.duration.show()
        table.attach(self.duration, 1, 2, 1, 2)

class Simulation(object):
    def __init__(self, parent):
        self.parent = parent
        self.simulator = None

        self.transforms = []
        self.current = 0
        self.speed = 1

        self.running = False
        self._do_stop = False

    def simulate(self, params, numbots, duration, time_step=1/30.0):
        context = cl.create_some_context()
        queue = cl.CommandQueue(context)

        simulator = physics.Simulator(context, queue, num_worlds=1, num_robots=numbots)
        simulator.set_ann_parameters(0, physics.ANNParametersArray.load(params))
        simulator.init_worlds(1.2)
        gobject.idle_add(self._simulator_step, simulator, duration)

    def _simulator_step(self, simulator, duration):
        if (simulator.clock) >= duration:
            self.parent.scene.fitness = None
            self.parent.scene.real_clock, self.parent.scene.transforms = self.transforms[0]
            self.current = 0
            return

        simulator.step()

        transforms = simulator.get_transforms()
        fitness = simulator.get_fitness()[0]
        clock = simulator.clock

        self.transforms.append((clock, transforms))
        self.parent.scene.transforms = transforms
        self.parent.scene.real_clock = simulator.clock
        self.parent.scene.fitness = fitness

        gobject.idle_add(self._simulator_step, simulator, duration)

    def start(self):
        self._do_stop = False

        if (self.current+1) >= len(self.transforms):
            self.running = False
            return

        if not self.running:
            self.running = True

            delay_time = (self.transforms[self.current+1][0] - self.transforms[self.current][0]) / self.speed
            gobject.timeout_add(int(delay_time * 1000), self._run)

    def stop(self):
        self._do_stop = True

    def _run(self):
        if self._do_stop:
            self.running = False
            return

        self.step()

        if (self.current+1) >= len(self.transforms):
            self.running = False
            return

        delay_time = (self.transforms[self.current+1][0] - self.transforms[self.current][0]) / self.speed
        gobject.timeout_add(int(delay_time * 1000), self._run)

    def step(self):
        if (self.current+1) >= len(self.transforms):
            return

        self.parent.scene.real_clock, self.parent.scene.transforms = self.transforms[self.current+1]
        self.parent.scene.speed = self.speed
        self.current += 1
