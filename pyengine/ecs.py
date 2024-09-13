from dataclasses import dataclass
import pygame
from typing import Tuple, Optional
from pprint import pprint
import cProfile


####################
#      BITWISE     #
####################
class _Bitset(int):
    def __repr__(self):
        return f"Comp-{_cm.id_to_component(self).__name__} [{int(self)}]"

    def __or__(self, other):
        return type(self)(super().__or__(other))

    def set(self, bit):
        return _Bitset(self | (1 << bit))

    def get_parts(self):
        return [2 ** i for i in range(self.bit_length()) if ((1 << i) & self)]


####################
#     ENTITIES    #
####################
class ComponentNotRegisteredError(Exception):
    pass


def get_archetype_id(comp_types):
    archetype_id = _Archetype()
    for comp_type in comp_types:
        try:
            archetype_id |= _cm.component_ids[comp_type]
        except KeyError:
            raise ComponentNotRegisteredError(f"Component of type {comp_type} has been tried to get involved in a system without its ID being initialized. Call register_components() on the component first.")
    return archetype_id


def create_entity(*comp_objects):
    # create new component id if component is unknown
    comp_types = [type(comp_obj) for comp_obj in comp_objects]
    archetype_id = get_archetype_id(comp_types)
    comp_ids = []
    for comp_type, comp_obj in zip(comp_types, comp_objects):
        # register component
        if comp_type not in _cm.component_ids:
            register_components(comp_type)
        # add possible new entry to the component -> archetype hashmap
        comp_id = _cm.component_ids[comp_type]
        comp_ids.append(comp_id)
        if comp_id not in _cm.archetypes_of_components:
            _cm.archetypes_of_components[comp_id] = set()
        if archetype_id not in _cm.archetypes_of_components[comp_id]:
            _cm.archetypes_of_components[comp_id].add(archetype_id)
    # update the archetype dict of dict of list after potentially registering new component
    if archetype_id not in _cm.archetype_pool:
        # archetype not not exist yet
        _cm.archetype_pool[archetype_id] = {}
    # append the components in the archetype key (as values)
    for comp_type, comp_id, comp_obj in zip(comp_types, comp_ids, comp_objects):
        if comp_id not in _cm.archetype_pool[archetype_id]:
            _cm.archetype_pool[archetype_id][comp_id] = [comp_obj]
        else:
            _cm.archetype_pool[archetype_id][comp_id].append(comp_obj)


####################
#    COMPONENTS    #
####################
def component(comp):
    register_components(comp)
    return comp


def register_components(*comp_types):
    for comp_type in comp_types:
        bitset = _Bitset().set(_cm.next_component_shift)
        _cm.component_ids[comp_type] = bitset
        _cm.next_component_shift += 1


class _Archetype(_Bitset):
    def __repr__(self):
        return f"Arch-{int(self )} [{", ".join(str(x) for x in self.get_parts())}]"


class _ComponentManager:
    def __init__(self):
        self.archetype_pool = {}
        self.component_ids = {}
        self.next_component_shift = 0
        self.archetypes_of_components = {}
    
    def id_to_component(self, id_):
        return {v: k for k, v in self.component_ids.items()}[int(id_)]


_cm = _ComponentManager()


####################
#      SYSTEMS     #
####################
def system(*component_types):
    def inner(system_type):
        def get_components(self):
            ret = []
            for arch in final_archetypes:
                for composite_comp_objects in zip(*(_cm.archetype_pool[arch][_cm.component_ids[comp_type]] for comp_type in component_types)):
                    ret.append(composite_comp_objects)
            return ret

        all_sets = []
        for comp_type in component_types:
            all_sets.append(_cm.archetypes_of_components[_cm.component_ids[comp_type]])
        final_archetypes = set.intersection(*all_sets)
        system_type.archetypes = final_archetypes
        system_type.get_components = get_components
        return system_type

    return inner


# component creation
@component
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
@dataclass
class Enemy:
    speed: float


@component
@dataclass
class Friendly:
    speed: float


@component
class Surface:
    def __init__(self, width: int, height: int, color: Tuple[int, int, int, Optional[int]]):
        self.width = width
        self.height = height
        self.surf = pygame.Surface((self.width, self.height))
        self.surf.fill(color)


@component
@dataclass
class Moveable:
    mult: int
    left: int = pygame.K_a
    right: int = pygame.K_d
    up: int = pygame.K_w
    down: int = pygame.K_s


# entity creation
create_entity(
    Surface(40, 30, (230, 74, 141)),
    Position((30, 30)),
    Enemy(10),
)

create_entity(
    Surface(80, 30, (45, 180, 17)),
    Position((80, 4)),
    Friendly(20),
    Moveable(5),
)


# system creation
@system(Surface, Position)
class RenderSystem:
    def process(self):
        for surf, pos in self.get_components():
            WIN.blit(surf.surf, pos)


@system(Position, Moveable)
class PhysicsSystem:
    def process(self):
        keys = pygame.key.get_pressed()
        for pos, moveable in self.get_components():
            if keys[moveable.left]:
                pos.x -= moveable.mult
            if keys[moveable.right]:
                pos.x += moveable.mult
            if keys[moveable.up]:
                pos.y -= moveable.mult
            if keys[moveable.down]:
                pos.y += moveable.mult


render_system = RenderSystem()
physics_system = PhysicsSystem()
WIN = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()


def main():
    # main loop

    while True:
        clock.tick(120)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    exit()
        
        WIN.fill((102, 120, 120))

        render_system.process()
        physics_system.process()

        pygame.display.flip()


cProfile.run("main()", sort="cumtime")
