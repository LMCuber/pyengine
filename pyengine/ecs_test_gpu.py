import pygame
from pygame._sdl2.video import Window, Renderer, Texture
import sys
from random import uniform as randf, randint as rand
from ecs import *
import cProfile
from pygame.time import get_ticks as ticks
from math import sin
from dataclasses import dataclass as component
# import numpy as np
# import esper


pygame.init()

WIDTH, HEIGHT = 1000, 800
win = Window(size=(WIDTH, HEIGHT))
ren = Renderer(win)
font = pygame.font.SysFont("Courier New", 20)
clock = pygame.time.Clock()


# @component
class Position(list):
    pass


# @component
class Velocity(list):
    pass


# @component
class Surface:
    def __init__(self, size, color):
        self.surf = pygame.Surface(size)
        self.surf.fill(color)
        self.tex = Texture.from_surface(ren, self.surf)
        self.offset = ticks()
    

def blit(*args):
    ren.blit(*args)


class PhysicsSystem:
    def __init__(self):
        # super().__init__(cache=True)
        # self.operates(Position, Velocity, Surface)
        pass
    
    def process(self):
        for ent_id, chunk, (pos, vel, surf) in get_components(Position, Velocity, Surface):
            pos[0] += vel[0]
            pos[1] += vel[1]
            if pos[0] < 0:
                vel[0] *= -1
                pos[0] = 0
            elif pos[0] > WIDTH:
                vel[0] *= -1
                pos[0] = WIDTH
            elif pos[1] < 0:
                vel[1] *= -1
                pos[1] = 0
            elif pos[1] > HEIGHT:
                vel[1] *= -1
                pos[1] = HEIGHT
            
            # ren.blit(surf.tex, pygame.Rect(*pos, 30, 30))
            blit(surf.tex, pygame.Rect(*pos, 30, 30))

        # WIN.blits(blits)


physics_system = PhysicsSystem()


num_entities = 2000
for i in range(num_entities):
    create_entity(
        Position((WIDTH / 2, HEIGHT / 2)),
        Velocity((randf(-3, 3), randf(-3, 3))),
        Surface((30, 30), [rand(0, 255) for _ in range(3)]),
        chunk=None
    )


def main():
    last = ticks()
    running = __name__ == "__main__"

    # physics_system.cache_components()

    while running:
        clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        ren.draw_color = (120, 120, 120)
        ren.clear()
        
        physics_system.process()
            
        fps = Texture.from_surface(ren, font.render(f"{int(clock.get_fps())}, {num_entities} entities", True, (230, 230, 230)))
        ren.blit(fps, pygame.Rect(5, 5, 400, 50))

        # pygame.display.update()
        ren.present()

        if ticks() - last >= 5000:
            running = False

    pygame.quit()
    sys.exit()


nump = False
cProfile.run(f"main()", sort="cumtime")
# main(ecs=ecs)
