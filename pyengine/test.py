import pygame
import sys
from random import uniform as randf, randint as rand
from ecs import *
import cProfile


pygame.init()
WIDTH, HEIGHT = 1000, 800
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
black_bar = pygame.Surface((WIDTH, 50))
black_bar.set_alpha(140)
font = pygame.font.SysFont("Courier New", 30)
clock = pygame.time.Clock()


class Player:
    def __init__(self):
        self.image = pygame.Surface((30, 30))
        self.image.fill([rand(0, 255) for _ in range(3)])
        self.pos = [WIDTH / 2, HEIGHT / 2]
        self.vel = [randf(-3, 3), randf(-3, 3)]
    
    def move(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        if self.pos[0] < 0:
            self.vel[0] *= -1
            self.pos[0] = 0
        elif self.pos[0] > WIDTH:
            self.vel[0] *= -1
            self.pos[0] = WIDTH
        elif self.pos[1] < 0:
            self.vel[1] *= -1
            self.pos[1] = 0
        elif self.pos[1] > HEIGHT:
            self.vel[1] *= -1
            self.pos[1] = HEIGHT
    
    def draw(self):
        WIN.blit(self.image, self.pos)
    
    def update(self):
       self.move()
       self.draw()


all_players = []
num_entities = 10000
for i in range(num_entities):
    all_players.append(Player())

@component
class Position(list):
    pass


@component
class Velocity(list):
    pass


@component
class Surface:
    def __init__(self, size, color):
        self.surf = pygame.Surface(size)
        self.surf.fill(color)


for i in range(num_entities):
    create_entity(
        Position((WIDTH / 2, HEIGHT / 2)),
        Velocity((randf(-3, 3), randf(-3, 3))),
        Surface((30, 30), [rand(0, 255) for _ in range(3)])
    )


@system(Position, Velocity, Surface)
class PhysicsSystem:
    def __init__(self):
        self.set_cache(True)

    def process(self):
        for pos, vel, surf in self.get_components():
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
            WIN.blit(surf.surf, pos)

physics_system = PhysicsSystem()


def main():
    running = __name__ == "__main__"
    while running:
        clock.tick(1000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        WIN.fill((120, 120, 120))

        # for player in all_players:
        #     player.update()
        
        physics_system.process()
        
        WIN.blit(black_bar, (0, 0))
        fps = font.render(f"{int(clock.get_fps())}, {num_entities} entities", True, (230, 230, 230))
        WIN.blit(fps, (5, 5))

        pygame.display.update()

    pygame.quit()
    sys.exit()


cProfile.run("main()", sort="cumtime")