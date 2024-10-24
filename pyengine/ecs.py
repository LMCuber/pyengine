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
        # comp_id = _cm.component_ids[comp_type]
        # comp_ids.append(comp_id)
        if comp_type not in _cm.archetypes_of_components:
            _cm.archetypes_of_components[comp_type] = set()
        if archetype_id not in _cm.archetypes_of_components[comp_type]:
            _cm.archetypes_of_components[comp_type].add(archetype_id)
    # update the archetype dict of dict of list after potentially registering new component
    if archetype_id not in _cm.archetype_pool:
        # archetype not not exist yet
        _cm.archetype_pool[archetype_id] = {}
    # append the components in the archetype key (as values)
    for comp_type, comp_obj in zip(comp_types, comp_objects):
        if comp_type not in _cm.archetype_pool[archetype_id]:
            _cm.archetype_pool[archetype_id][comp_type] = [comp_obj]
        else:
            _cm.archetype_pool[archetype_id][comp_type].append(comp_obj)


####################
#    COMPONENTS    #
####################
def component(comp_type):
    register_components(comp_type)
    return comp_type


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
        def get_components(self, cache=True):
            if self.component_cache is not None and cache:
                return self.component_cache
            else:  # for clarity
                ret = []
                for arch in final_archetypes:
                    for index, composite_comp_objects in enumerate(zip(*(_cm.archetype_pool[arch][comp_type] for comp_type in component_types))):
                        ret.append((composite_comp_objects if len(composite_comp_objects) > 1 else composite_comp_objects[0]))
                self.component_cache = ret
                return ret

        all_sets = []
        for comp_type in component_types:
            all_sets.append(_cm.archetypes_of_components[comp_type])
        final_archetypes = set.intersection(*all_sets)
        system_type.archetypes = final_archetypes
        system_type.get_components = get_components
        system_type.component_cache = None
        return system_type

    return inner
