"""
pygenesis/__init__.py

PyGenesis: A Python library for building and experimenting with custom neural networks from scratch
Copyright (C) 2023  Austin Berrio

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"When did the future switch from being a promise to being a threat?"
    - Chuck Palahniuk, Invisible Monsters
"""
import logging

__version__ = "0.0.1"
__name__ = "pygenesis"
__agent__ = "teleprint-me/pygenesis"
__source__ = "https://github.com/teleprint-me/pygenesis"
__author__ = "Austin Berrio"
__email__ = "aberrio@teleprint.me"

# Set logging configuration
# NOTE: Can be overridden on a script-by-script basis
logging_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s"
logging.basicConfig(format=logging_format, level=logging.DEBUG)
