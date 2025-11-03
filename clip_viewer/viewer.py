import pygame as pg
import numpy as np

from viztools.drawable.lines import Lines
from viztools.viewer import Viewer


class LineViewer(Viewer):
    def __init__(self, points):
        super().__init__(drag_mouse_button=2)
        self.lines = Lines(
            points,
            color=np.array([0, 255, 0, 50]),
        )

    def tick(self, delta_time: float):
        self.update_drawables([self.lines])

    def render(self):
        self.render_coordinate_system(draw_numbers=True)
        self.render_drawables([self.lines])

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)


def main():
    viewer = LineViewer()
    viewer.run()
