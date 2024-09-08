from dataclasses import dataclass as component
import pygame
from typing import Tuple, Optional
from pprint import pprint


####################
#     ENTITIES    #
####################
class _EntityManager:
    def __init__(self):
        self.entity_masks = []
    
    def add_entity(self):
        pass


_entity_manager = _EntityManager()


class _Entity:
    def __init__(self, mask):
        self.mask = mask


def create_entity(*components):
    for comp_obj in components:
        # component
        comp_type = type(comp_obj)
        _component_manager.add_component(comp_type, comp_obj)
        print(_component_manager.component_ids[comp_type])

        
    # entity
    # _entity_manager.add_entity()


####################
#    COMPONENTS    #
####################
class _ComponentManager:
    def __init__(self):
        self.next_component_id = 0
        self.component_pool = {}
        self.component_ids = {}
    
    def add_component(self, comp_type, comp_obj):
        if comp_type not in self.component_pool:
            self.component_pool[comp_type] = []
            self.component_ids[comp_type] = self.next_component_id
            self.next_component_id += 1
        self.component_pool[comp_type].append(comp_obj)


_component_manager = _ComponentManager()


####################
#      SYSTEMS     #
####################
def system(*component_types):
    def inner(system_type):
        #
        def get_components(self):
            return list(zip(*[_component_manager.component_pool[comp_type] for comp_type in component_types]))

        system_type.get_components = get_components
        return system_type

    return inner


# component creation
class Position(list):
    @property
    def x(self):
        return self[0]
    
    @x.setter
    def x(self, value):
        self[0] = value
    
    @property
    def y(self):
        return self[1]
    
    @y.setter
    def y(self, value):
        self[1] = value


@component
class Image:
    image: pygame.Surface


@component
class Keys:
    pass


class Surface:
    def __init__(self, width: int, height: int, color: Tuple[int, int, int, Optional[int]]):
        self.width = width
        self.height = height
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill(color)

    
# entity creation
entity = create_entity(
    Position((10, 10)),
    Surface(40, 30, (230, 74, 141))
)


# system creation
@system(Surface, Position)
class RenderSystem:
    def process(self):
        for surf, pos in self.get_components():
            WIN.blit(surf.surf, (pos.x, pos.y))


# main loop
render_system = RenderSystem()
WIN = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
while True:
    clock.tick(120)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    
    WIN.fill((102, 120, 120))

    render_system.process()

    pygame.display.flip()
