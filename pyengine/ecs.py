from dataclasses import dataclass as component
import pygame
from typing import Tuple, Optional
from pprint import pprint


####################
#      CLASSES     #
####################
class Bitset(int):
    def set(self, bit, value):
        self = Bitset(value | (1 << bit))


####################
#     ENTITIES    #
####################
def create_entity(*components):
    archetype = tuple(type(comp_obj) for comp_obj in components)
    # create new component id if component is unknown
    for comp_obj in components:
        comp_type = type(comp_obj)
        if comp_type not in _component_manager.component_ids:
            _component_manager.register_component(comp_type)
    # update the archetype dict of dict of list
    if archetype not in _component_manager.archetype_pool:
        # new archetype key
        _component_manager.archetype_pool[archetype] = {}
        for comp_obj in components:
            for comp_type, comp_obj in zip(archetype, components):
                # append to existing archetype
                _component_manager.archetype_pool[archetype][comp_type] = comp_obj
    else:
        # existing archetype
        for comp_type, comp_obj in zip(archetype, components):
            _component_manager.archetype_pool[archetype][comp_type].append(comp_obj)


####################
#    COMPONENTS    #
####################
class _ComponentManager:
    def __init__(self):
        self.archetype_pool: dict[dict[list]] = {}
        self.component_ids = {}
        self.next_component_shift = 0

    def register_component(self, comp_type):
        bitset = Bitset()
        bitset.set(self.next_component_shift, 1)
        _component_manager.component_ids[comp_type] = bitset
        

_component_manager = _ComponentManager()


####################
#      SYSTEMS     #
####################
def system(*component_types):
    def inner(system_type):
        #
        def get_components(self):
            print("-"*50)
            components = [_component_manager.archetype_pool[self.archetype][comp_type] for comp_type in component_types]
            pprint(components)
            print("-"*50)
            return components

        system_type.archetype = component_types
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
class Key:
    up: int
    down: int
    left: int
    right: int
    mult: int


class Surface:
    def __init__(self, width: int, height: int, color: Tuple[int, int, int, Optional[int]]):
        self.width = width
        self.height = height
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill(color)


# entity creation
create_entity(
    Surface(40, 30, (230, 74, 141)),
    Position((10, 10)),
)


# system creation
@system(Surface, Position)
class RenderSystem:
    def process(self):
        for surf, pos in self.get_components():
            WIN.blit(surf.surf, (pos.x, pos.y))


@system(Key, Position)
class KeySystem:
    def process(self):
        k = pygame.key.get_pressed()
        for key, position in self.get_components():
            if k[key.up]:
                position.y -= key.mult
            if k[key.down]:
                position.y += key.mult
            if k[key.left]:
                position.x -= key.mult
            if k[key.right]:
                position.x += key.mult


render_system = RenderSystem()

# main loop
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