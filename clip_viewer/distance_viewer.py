import pygame as pg
import numpy as np
from viztools.drawable.draw_utils import AnkerType

from viztools.drawable.lines import Lines
from viztools.drawable.image import Image
from viztools.drawable.overlay_text import OverlayText
from viztools.viewer import Viewer


COLORS = [
    np.array([255, 0, 0, 50]),
    np.array([0, 255, 0, 50]),
    np.array([80, 80, 255, 50]),
    np.array([255, 255, 0, 50]),
    np.array([255, 0, 255, 50]),
    np.array([0, 255, 255, 50]),
]


class DistanceViewer(Viewer):
    def __init__(self, distances, images, queries):
        super().__init__(drag_mouse_button=2)
        x_positions = np.arange(len(distances)) * 0.04
        self.lines = [
            Lines(np.stack([x_positions, pos], axis=1), color=color) for pos, color in zip(distances.T, COLORS)
        ]
        positions = np.stack([x_positions, distances[:, 0]], axis=1)
        self.images = [
            Image(
                img, pos, size=0.003, offset=np.array([0, -0.25]),
                offset_color=np.array([180, 180, 180]), anker_type=AnkerType.TOP
            ) for img, pos in zip(images, positions)
        ]
        self.active_images = []
        query_positions = np.stack([np.zeros(distances.shape[1]), distances[0]], axis=1)
        self.queries = [
            OverlayText(query, pos, color=color)
            for query, pos, color in zip(queries, query_positions, COLORS)
        ]

    def tick(self, delta_time: float):
        self.update_drawables(self.lines)

    def render(self):
        self.render_coordinate_system(draw_numbers=True)
        self.render_drawables(self.lines)
        activate_images = [self.images[index] for index in self.active_images]
        self.render_drawables(activate_images)
        self.render_drawables(self.queries)

    def handle_event(self, event: pg.event.Event):
        super().handle_event(event)
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            self.active_images = self.lines[0].clicked_points(event, self.coordinate_system)
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
