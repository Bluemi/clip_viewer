import pygame as pg
import numpy as np
from viztools.drawable.draw_utils import AnkerType

from viztools.drawable.lines import Lines
from viztools.drawable.image import Image
from viztools.viewer import Viewer


class LineViewer(Viewer):
    def __init__(self, points, images):
        super().__init__(drag_mouse_button=2)
        self.lines = Lines(
            points,
            color=np.array([0, 255, 0, 50]),
        )
        self.images = [Image(img, pos, anker_type=AnkerType.TOP_LEFT) for img, pos in zip(images, points)]
        self.active_images = []

    def tick(self, delta_time: float):
        self.update_drawables([self.lines])

    def render(self):
        self.render_coordinate_system(draw_numbers=True)
        self.render_drawables([self.lines])
        activate_images = [self.images[index] for index in self.active_images]
        self.render_drawables(activate_images)

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.active_images = self.lines.clicked_points(event, self.coordinate_system)
            self.render_needed = True
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RIGHT:
                self.change_active_image(1)
            elif event.key == pg.K_LEFT:
                self.change_active_image(-1)
            elif event.key == pg.K_0:
                self.active_images = [0]
            self.render_needed = True

    def change_active_image(self, change: int = 1):
        self.active_images = [ai + change for ai in self.active_images if 0 <= ai + change < len(self.images)]
