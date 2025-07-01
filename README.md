# PyEngine

A python _package_ I concocted for use in complex 2D video games. It consists of the following _modules_:
* `basics`: basic functions that Python doesn't have in its standard library, such as:
    * `get_clipboard` get the current user clipboard
    * `delay`: a non-blocking function that delays the execution of a callback, comparable to JavaScript's `setTimeout`
    * `rand_alien_name`: generates a very shitty alien name. Use with caution.\
Has too much random shit that doesn't need to be there, so it's still under refactorment.
* `pgbasics`: extra `pygame` functionality for your game, such as:
    * `imgload`: an extended image loading function that can parse spritesheets and scale the images
    * `rot_pivot`: rotating an image and a rect around a pivot point instead of the default center
    * `Crystal`: a 3D mesh + texture class capable of rendering 3D objects
* `pgshaders`: a wrapper around the `moderngl` pipeline for shaders in `pygame`. Consists only of the `ModernglShader` class.
* `pgwidgets`: `pygame` UI widgets, such as a `Button`, `Slider` and `Checkbox`.
* `pilbasics`: currently abandoned `pillow` helper functions, such as `get_isometric_3d`.
* `ecs`: a chunk-based _Entity Component System_ based on archetypes. 
