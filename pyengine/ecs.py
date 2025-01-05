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
        return f"Component-{_cm.id_to_component(self).__name__} [{int(self)}]"

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


def create_entity(*comp_objects, chunk):
    # init vars
    comp_types = [type(comp_obj) for comp_obj in comp_objects]
    archetype_id = get_archetype_id(comp_types)
    comp_ids = []
    # register the components
    for comp_type, comp_obj in zip(comp_types, comp_objects):
        # register component
        if comp_type not in _cm.component_ids:
            register_components(comp_type)
        # add possible new entry to the [component -> archetype] hashmap
        if comp_type not in _cm.archetypes_of_components:
            _cm.archetypes_of_components[comp_type] = set()
        _cm.archetypes_of_components[comp_type].add(archetype_id)
    # create new ecs for a chunk that doesn't exits yet
    if chunk not in _cm.archetype_pool:
        _cm.archetype_pool[chunk] = {}
    # update the archetype dict of dict of list when registering new component
    if archetype_id not in _cm.archetype_pool[chunk]:
        # archetype not not exist yet
        _cm.archetype_pool[chunk][archetype_id] = {}
    # append the components in the archetype key
    for comp_type, comp_obj in zip(comp_types, comp_objects):
        if comp_type not in _cm.archetype_pool[chunk][archetype_id]:
            _cm.archetype_pool[chunk][archetype_id][comp_type] = [comp_obj]
        else:
            _cm.archetype_pool[chunk][archetype_id][comp_type].append(comp_obj)
    # update the archetypes for the intersections
    for comp_type in comp_types:
        for (system, index) in _sm.systems_of_components.get(comp_type, []):
            system.init_intersection(index)


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
        return f"Archetype-{int(self)}([{", ".join(str(x) for x in self.get_parts())}] = {bin(self).removeprefix("0b")})"


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
def process_systems():
    for system in _sm.iter_systems:
        system.process()


def add_system(system):
    _sm.iter_systems.append(system)


class _SystemManager():
    def __init__(self):
        self.systems_of_components = {}
        self.iter_systems = {}


_sm = _SystemManager()


# system decorator for a class
class System:
    def __init__(self):
        pass

    pass


def system(system_type):
    def set_cache(self, tof):
        if tof and not hasattr(self, "cache"):
            self.component_cache = []
            self.cache_updated = True
        self.cache = tof

    def get_components(self, index, chunks):
        # initialize returned components
        ret = []
        # the component_types that we need
        component_types = self._archetypes[index]
        for chunk in chunks:
            # check if chunk has any entity entries
            if chunk in _cm.archetype_pool:
                for arch in self._intersection_of_archetypes[index]:
                    # extend the list with the components of the archetype
                    # in the fashion (eid, chunk, (comp1, comp2, comp3, ... compN))
                    ret.extend([(eid, chunk, comps) for (eid, comps) in enumerate(zip(*(_cm.archetype_pool[chunk][arch][comp_type] for comp_type in component_types if arch in _cm.archetype_pool[chunk])))])
        if self.cache:
            self.component_cache = ret
        return ret
    
    def operates(self, *comp_types):
        # tells me which archetypes the system operates on
        # gets accessed using an index
        self._archetypes.append(comp_types)
        for comp_type in comp_types:
            # links the component type with the system that it operates on,
            # as well as the index (because systems may have multiple archetypes)
            iden = (self, len(self._archetypes) - 1)
            if comp_type in _sm.systems_of_components:
                _sm.systems_of_components[comp_type].append(iden)
            else:
                _sm.systems_of_components[comp_type] = [iden]
    
    def init_intersection(self, index):
        # get the intersection of all combinations of archetypes that all the components are in.
        # this also has to be reinitialized when a new entity is created, since
        # since the components are now in (pot.) more archetypes
        # [to reinitialize, check each system that has any of the created component types and call init_intersection]
        
        # iterate over the archetypes
        all_sets = []
        for comp_type in self._archetypes[index]:
            if comp_type in _cm.archetypes_of_components:
                all_sets.append(_cm.archetypes_of_components[comp_type])
            else:
                all_sets.append(set())
        # intersection because which archetype has at least all of the necessary component types, you know what I mean
        self._intersection_of_archetypes[index] = set.intersection(*all_sets) if all_sets else set()

    og_init = system_type.__init__

    def __init__(self, *args, **kwargs):
        # stores the archetypes the systems intends to operate on
        self._archetypes = []
        self._intersection_of_archetypes = {}
        # og_init may have any number of operates() calls
        og_init(self, *args, **kwargs)

    # instance methods
    system_type.get_components = get_components
    system_type.set_cache = set_cache
    system_type.__init__ = __init__
    system_type.init_intersection = init_intersection
    system_type.operates = operates
    # return
    return system_type
