from dataclasses import dataclass
from itertools import count
from typing import Type, Any
from pprint import pprint


_CompType = Type[Any]
_CompObj = Any
_EntityType = int
_ChunkType = Any  # can be anything set by the user


def has_component(ent_id, comp_type):
    return comp_type in _em.entities[ent_id]


def try_component(ent_id, comp_type):
    if comp_type in _em.entities[ent_id]:
        return _em.entities[ent_id][comp_type]
    return False


def create_entity(*comp_objects, chunk=None):
    ent_id = next(_em.counter)
    _em.entities[ent_id] = {}

    for comp_obj in comp_objects:
        comp_type = type(comp_obj)

        if chunk not in _cm.components:
            _cm.components[chunk] = {}

        if comp_type not in _cm.components[chunk]:
            _cm.components[chunk][comp_type] = set()
        _cm.components[chunk][comp_type].add(ent_id)

        _em.entities[ent_id][comp_type] = comp_obj
    
    clear_cache()


def delete_entity(ent_id, chunk):
    # if off-by-one error (especially when nested iterations), do nothing, entity has probabel already been removed
    if ent_id not in _em.entities:
        return

    for comp_type in _em.entities[ent_id]:
        # for each component entity was part of, entity id gets removed from the component list
        _cm.components[chunk][comp_type].discard(ent_id)
        if not _cm.components[chunk][comp_type]:
            del _cm.components[chunk][comp_type]
    
    # delete the actual entity data
    del _em.entities[ent_id]

    clear_cache()


def relocate_entity(ent_id, src_chunk, dest_chunk):
    # deletes and creates in one (can call other functions since it doesn't get called that often)
    # save the entity data before deleting
    comp_objects = list(_em.entities[ent_id].values())
    # delete
    delete_entity(ent_id, src_chunk)
    # recreate
    create_entity(*comp_objects, chunk=dest_chunk)


def get_components(*comp_types, chunks=(None,)):
    """
    Returns components from all given chunks and checks for cache first.
    """
    # for each chunk, check if it has cache or not
    ret = []
    for chunk in chunks:
        try:
            # try to get the components from cache if no new entities were added
            ret.extend(_cm.cache[(comp_types, chunk)])
        except KeyError:
            # there was a cache flush so have to query again. Saves to cache and returns.
            ret.extend(_cm.cache.setdefault((comp_types, chunk), list(_get_components(*comp_types, chunk=chunk))))
            
    return ret
    

def _get_components(*comp_types, chunk):
    """
    Returns the components from a single chunk regardless of cache (internal function).
    """
    try:
        intersected_entities = set.intersection(*[_cm.components[chunk][comp_type] for comp_type in comp_types])
        for entity in intersected_entities:
            yield entity, chunk, [_em.entities[entity][comp_type] for comp_type in comp_types]
    except KeyError:
        pass


def clear_cache():
    """
    cache caches 2D: tuple[comp types], chunk_index
    cache needs to clear when:
        - new entity is created
        - an entity is deleted
        - an entity relocates to a different chunk (combination of creation and deletion)
    can be called manually when needed
    """
    _cm.cache.clear()


class _EntityManager:
    def __init__(self):
        self.counter = count(start=0, step=1)
        self.entities: dict[_EntityType, dict[_CompType, _CompObj]] = {}


class _ComponentManager:
    def __init__(self):
        self.cache: dict[tuple[tuple[_CompType], _ChunkType], list[_EntityType]] = {}
        self.components: dict[_ChunkType, dict[_CompType, list[_EntityType]]] = {}


_em = _EntityManager()
_cm = _ComponentManager()


class System:
    def set_cache(self, tof):
        pass
