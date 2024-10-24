import pygame
import moderngl
import array


def get_shader(vertex_path, fragment_path, data=None, vertex_in=("vert", "texcoord")):
    # context
    ctx = modernl.create_context()
    # quad buffer
    if data is None:
        data = array("f", [
            -1.0, 1.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
            1.0, -1.0, 1.0, 1.0,
        ])
    quad_buffer = ctx.buffer(data=data)
    # shaders
    with open(vertex_path, "r") as f:
        vertex_shader = f.read()
    with open(fragment_path, "r") as f:
        fragment_shader = f.read()
    # program
    program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    shader = ctx.vertex_array(program, [(quad_buffer, "2f 2f", *vertex_in)])
    
