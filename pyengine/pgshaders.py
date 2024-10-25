import moderngl
from array import array
from pathlib import Path


# moderngl init
class ModernglShader:
    def __init__(self, vertex_shader, fragment_shader):
        self.ctx = moderngl.create_context()
        self.quad_buffer = self.ctx.buffer(data=array("f", [
            # pos (x, y), uv (x, y) (mapping the vertices to the texture space, we flip the uv's on the y-axis because pygame uses inverted cartesian coordinates)
            -1.0, 1.0, 0.0, 0.0,  # topleft
            1.0, 1.0, 1, 0.0,  # topright
            -1.0, -1.0, 0.0, 1,  # bottomleft
            1.0, -1.0, 1, 1,  # bottomright
        ]))
        # shaders (vertex & fragment)
        with open(vertex_shader, "r") as f:
            self.vert_shader = f.read()
        with open(fragment_shader, "r") as f:
            self.frag_shader = f.read()
        # moderngl shit
        self.program = self.ctx.program(vertex_shader=self.vert_shader, fragment_shader=self.frag_shader)
        self.render_object = self.ctx.vertex_array(self.program, [(self.quad_buffer, "2f 2f", "vert", "texcoord")])
    
    def _surf_to_tex(self, surf):
        self.tex = self.ctx.texture(surf.get_size(), 4)
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex.swizzle = "BGRA"
        self.tex.write(surf.get_view("1"))
        return self.tex
    
    def send_surf(self, index, key, surf):
        tex = self._surf_to_tex(surf)
        tex.use(index)
        if key == "tex":
            self.tex = tex
        self.program[key] = index
    
    def send(self, key, value):
        self.program[key] = value
    
    def render(self, mode=moderngl.TRIANGLE_STRIP):
        self.render_object.render(mode=mode)

    def release(self):
        self.tex.release()
        