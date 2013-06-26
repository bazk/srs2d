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
Gateway to the viewer (instantiate the Main window).
"""

__author__ = "Eduardo L. Buratti <eburatti09@gmail.com>"
__date__ = "26 Jun 2013"

import gui.main
import logging

__log__ = logging.getLogger(__name__)

if __name__=="__main__":
    gui.main.Main()
