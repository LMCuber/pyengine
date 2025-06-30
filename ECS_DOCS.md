# PyEngine

A python _package_ I concocted for use in complex 2Dvideo games. It consists of the following _modules_:
* `basics`: basic functions that Python doesn't have in its standard library, such as:
    * `get_clipboard` get the current user clipboard
    * `delay`: a non-blocking function that delays the execution of a callback, comparable to JavaScript's `setTimeout`
    * `rand_alien_name`: generates a very shitty alien name. Use with caution.
Has too much random shit that doesn't need to be there, so it's still under refactorment.
* `pgbasics`: extra `pygame` functionality for your game, such as:
    * `imgload`: an extended image loading function that can parse spritesheets and scale the images
    * `rot_pivot`: rotating an image and a rect around a pivot point instead of the default center
    * `Crystal`: a 3D mesh + texture class capable of rendering 3D objects
* `pgshaders`: a wrapper around the `moderngl` pipeline for shaders in `pygame`. Consists only of the `ModernglShader` class.
* `pgwidgets`: `pygame` UI widgets, such as a `Button`, `Slider` and `Checkbox`.
* `pilbasics`: currently abandoned `pillow` helper functions, such as `get_isometric_3d`.
* `ecs`: a chunk-based _Entity Component System_ based on archetypes. 

# ECS Documentation
## What is an ECS?
An entity component system is a data-oriented way of storing information. It can be applied in a lot of fields, but we will be looking at it from a game development point of view.
Data-oriented means, in simple terms, that the data of the entities our game (all assets, properties, etc.) are stored in such a way in memory that that relevant data is stored next to each other in a linear fashion. So instead of storing all entities in an object oriented way like below:

```python
entities = [
    {"pos": (0, 0), "img": pygame.Surface((30, 30))},
    {"pos": (5, 10), "img": pygame.Surface((20, 10))},
    ...
]
```
we store the entities as lists of their _components_:

```
positions = [
    (0, 0),
    (5, 10),
    ...
]
images = [
    pygame.Surface((30, 30)),
    pygame.Surface((20, 10)),
    ...
]
```
If you're familiar with databases, this way of storing data is comparable to how relational databases store their data compared to a NoSQL databases such as MongoDB, which uses a form of JSON.

An entity component system consists of three main ideas: _entities_, _components_ and _systems_. A _chunk-based_ ecs simply means that it splits the world into smaller more manageable chunks, so it doesn't have to update off-screen entities. This is why the starting chunk of the entity has to be specified when creating one with `create_entity`.

## Components
Suppose we want to create an entity with a given position, velocity and a surface, and we want to render it to our `pygame` window.
Components are parts of an entity that can be seperable from each other, such as `position` and `image`. In `pyengine`, components are marked with the `component` decorator. The implementation of a simple `transform` class can be done as follows:
```python
@component
@dataclass
class Transform:
    pos: list[float, float]
    vel: list[float, float]
```
Note: it is recommended to use `dataclass` when working with _POD_ (plain-old-data classes - components that just have data and no initialization logic), as it makes code more readable that an elaborate `__init__` function.
As a counterexample, below is an example of an `image` component:

```python
@component
class Image:
    @classmethod
    def from_surface(size: tuple[int, int]):
        self = cls()
        self.surf = pygame.Surface(size)
    
    @classmethod
    def from_path(path: str):
        self = cls()
        self.surf = imgload(path)
```
However, as you can probably tell, when we want to create an entity, we can't just create `transform` and `image` components and hope for them to magically work together. We use the function `create_entity` for this:

```python
    create_entity(
        Transform([0, 0], [0.2, -4.5]),
        Image.from_path("assets/test.png"),
        chunk=None
    )
```
Note: `chunk=None` means by convention that the entity does not adhere to the divisions of space and will be updated regardless.

But now, suppose we want to increase the transform of the entity every frame. How do we process our entities, since we have lost all references to them immediately after their creation? This is where systems come in.

## Systems
Systems operate on components - it's as simple as that. You decorate your class with `system` and tell it which components it operates on. It is common to have multiple systems operating on two or even one component.

```python
@system
class RenderSystem:
    def __init__(self):
        self.operates(Transform, Image)
    
    def process(self):
        for ent, chunk, (tr, img) in self.get_components(0, chunks=None):
            # move the position by the velocity
            tr.pos[0] += tr.vel[0]
            tr.pos[1] += tr.vel[1]
            # render the image to the display
            screen.blit(img.img, tr.pos)
```
A few important things to note here:
* `get_components` is the crux here: it returns all entities that have _at least_ the `transform` and `image` components. So an enemy that also has the `health` component will still be rendered. The first argument is the operation index, but since we only have one operation (the Transform and Image pair), we input zero. The `chunks` parameters can be set by the user if only updates chunks need to update their entities, but in this example we will use `None`.
* `get_components` returns 3 values: the entity index, the chunk and a tuple of all the components. The first two can be ignored in this scenario, but can be relevant when deleting or relocating entities.

## Entities
Entities aren't a real thing: they are just the index that corresponds to a collection of components, such as (Transform, Image).
