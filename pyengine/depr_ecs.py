from __future__ import annotations
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

class _Archetype(_Bitset):
    def __repr__(self):
        return f"Archetype-{int(self)}([{", ".join(str(x) for x in self.get_parts())}] = {bin(self).removeprefix("0b")})"


class _EntityManager:
    def __init__(self):
        self.ent_id = 0
        self.ent_to_components: dict[_EntityType, tuple[_Archetype, int]] = {}
    
    def get_id(self):
        self.ent_id += 1
        return self.ent_id - 1

    def save_entity_bitmask(self, ent_id, arch):
        self.ent_to_components[ent_id] = (arch,)
    
    def create_entity(self, *comp_objects, chunk):
        ent_id = self.get_id()
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

        if "id" not in _cm.archetype_pool[chunk][archetype_id]:
            _cm.archetype_pool[chunk][archetype_id]["id"] = [ent_id]
        else:
            _cm.archetype_pool[chunk][archetype_id]["id"].append(ent_id)

        # update the archetypes for the intersections
        for comp_type in comp_types:
            for (system, index) in _sm.systems_of_components.get(comp_type, []):
                system.init_intersection(index)

        # update the entity -> component type bitmask
        _em.save_entity_bitmask(ent_id, archetype_id)

_em = _EntityManager()
create_entity = _em.create_entity


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


class System:
    def __init__(self, cache=True):
        self.set_cache(cache)
        # stores the archetypes the systems intends to operate on
        self._archetypes = []
        self._intersection_per_subsystem = {}

    def has_component(self, ent_id, comp_type):
        pass

    def set_cache(self, tof):
        if tof and not hasattr(self, "cache"):
            self.component_cache = []
            self.cache_updated = True
        self.cache = tof
    
    def delete(self, archetype, index, chunk):
        for comp_type in _cm.archetype_pool[chunk][archetype]:
            # when two entities are deletet in the same loop, the indices don't match up anymore, so we must shift the last one down by 1 for each deleted entity
            # HACK: this is jank but I haven't stumbled upon a bug yet
            test_i = index
            while test_i >= 0:
                try:
                    del _cm.archetype_pool[chunk][archetype][comp_type][test_i]
                except IndexError:
                    test_i -= 1
                    continue
                else:
                    break
    
    def relocate(self, src_chunk, archetype, ent_index, dest_chunk):
        # backup all components of this single entity
        # HACK: again, same hack as last time
        test_i = ent_index
        while test_i >= 0:
            try:
                entity_components = [comps[test_i] for comps in _cm.archetype_pool[src_chunk][archetype].values()]
            except IndexError:
                test_i -= 1
                continue
            else:
                break

        # delete the entity
        self.delete(archetype, ent_index, src_chunk)

        # # add the entity to the new destination chunk
        create_entity(
            *entity_components,
            chunk=dest_chunk
        )

    def get_components(self, subsystem_index, chunks):
        # initialize importancies
        ret = []

        # the component_types that we need
        component_types = self._archetypes[subsystem_index]
        for chunk in chunks:
            # check if chunk has any entities at all
            if chunk in _cm.archetype_pool:
                # XXX: what is this if statement below?
                if subsystem_index not in self._intersection_per_subsystem:
                    continue
                for arch in self._intersection_per_subsystem[subsystem_index]:
                    # extend the list with the components of the archetype
                    # in the fashion (ent_index, chunk, (ent_id, comp_1, comp_2, comp_3, ... comp_n))
                    ent_id = _cm.archetype_pool[chunk][arch]["id"]
                    ret.extend([
                        (
                            ent_index,
                            chunk,
                            arch,
                            ent_id,
                            comps,
                        )
                        for (ent_index, comps) in enumerate(
                            zip(*(
                                _cm.archetype_pool[chunk][arch][comp_type]
                                for comp_type in component_types
                                if arch in _cm.archetype_pool[chunk]
                            ))
                        )
                    ])
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
        self._intersection_per_subsystem[index] = set.intersection(*all_sets) if all_sets else set()
