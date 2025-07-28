import pygame
from pygame.time import get_ticks as ticks
import pymunk
from pygame.locals import *
from pygame._sdl2.video import Window, Renderer, Texture, Image
from pygame.math import Vector2, Vector3
from numpy import array
import pygame.gfxdraw
import pygame.midi
import typing
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_hls, hls_to_rgb
import cv2
from scipy.spatial import Delaunay
from dataclasses import dataclass
#
from .imports import *
from .basics import *
from .pilbasics import pil_to_pg


pygame.init()
pygame.midi.init()

# event costants
is_left_click = lambda event: event.type == pygame.MOUSEBUTTONDOWN and event.button == 1

# colors
BLACK =         (  0,   0,   0, 255)
BLAC =          (  1,   1,   1, 255)
ALMOST_BLACK =  ( 40,  40,  40, 255)
WHITE =         (255, 255, 255, 255)
SILVER =        (210, 210, 210, 255)
LIGHT_GRAY =    (180, 180, 180, 255)
GRAY =          (120, 120, 120, 255)
DARK_GRAY =     ( 80,  80,  80, 255)
WIDGET_GRAY =   (150, 150, 150, 255)
RED =           (255,   0,   0, 255)
DARK_RED =      ( 56,   0,   0, 255)
LIGHT_PINK =    (255, 247, 247, 255)
PINK_RED =      pygame.Color("#FF474C")
GREEN =         (  0, 255,   0, 255)
MOSS_GREEN =    ( 98, 138,  56, 255)
DARK_GREEN =    (  0,  70,   0, 255)
DARKISH_GREEN = (  0,  120,  0, 255)
LIGHT_GREEN =   (  0, 255,   0, 255)
SLIME_GREEN =   (101, 255,   0, 255)
MINT =          (186, 227, 209, 255)
TURQUOISE =     ( 64, 224, 208, 255)
AQUAMARINE =    ( 15,  99, 109, 255)
SLIMISH =       ( 88, 199, 151, 255)
YELLOW =        (255, 255,   0, 255)
YELLOW_ORANGE = (255, 174,  66, 255)
SKIN_COLOR =    (255, 219, 172, 255)
GOLD =          (255, 214,   0, 255)
CYAN =          (  0, 255, 255, 255)
BLUE =          (  0,  0,  255, 255)
POWDER_BLUE =   (176, 224, 230, 255)
WATER_BLUE =    ( 17, 130, 177, 255)
SKY_BLUE =      (220, 248, 255, 255)
LIGHT_BLUE =    (137, 209, 254, 255)
SMOKE_BLUE =    (154, 203, 255, 255)
DARK_BLUE =     (  0,   0,  50, 255)
NAVY_BLUE =     ( 32,  42,  68, 255)
LAPIS_BLUE =    ( 49,  53,  92, 225)
BLUE =          (  0,   0, 200, 255)
PURPLE =        (153,  50, 204, 255)
DARK_PURPLE =   ( 48,  25,  52, 255)
ORANGE =        (255,  94,  19, 255)
BROWN =         (125,  70,   0, 255)
DARKISH_BROWN = ( 87,  45,   7, 255)
LIGHT_BROWN =   (149,  85,   0, 255)
DARK_BROWN =    (101,  67,  33, 255)
PINK =          (255, 192, 203, 255)
CREAM =         pygame.Color("#F8F0C6")
ALPHA_GRAY =    lambda x: (80, 80, 80, x)

# other constants
INF = "\u221e"  # infinity symbol (unicode)
orthogonal_projection_matrix = array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])
resolutions = [(640 * m, 360 * m) for m in range(1, 6)]


# CPU definitions (can be overwritten later)
def fill_display(display, color):
    display.fill(color)


draw_line = pygame.draw.line
draw_rect = pygame.draw.rect
draw_aacircle = pygame.draw.aacircle
fill_rect = pygame.draw.rect
draw_quad = pygame.draw.polygon
fill_quad = pygame.draw.polygon
draw_triangle = pygame.draw.polygon
fill_triangle = pygame.draw.polygon
rotate_ip = lambda_none


def flip_x(surf):
    return pygame.transform.flip(surf, True, False)


def scale_by(surf, mult):
    return T(pygame.transform.scale_by(surf, mult))


class CImage(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width = self.texture.width
        self.height = self.texture.height

    def get_rect(self, *args, **kwargs):
        return Texture.get_rect(self.texture, *args, **kwargs)


class SurfaceBuilder:
    def __init__(self, *args, **kwargs):
        self.surf = pygame.Surface(*args, **kwargs)
    
    @classmethod
    def from_image(cls, img):
        self = cls()
        self.surf = img
        return self
    
    def fill(self, *args, **kwargs):
        self.surf.fill(*args, **kwargs)
        return self
    
    def set_alpha(self, *args, **kwargs):
        self.surf.set_alpha(*args, **kwargs)
        return self
    
    def build(self):
        return self.surf
    

class _Global:
    def __init__(self):
        self.hwaccel = False
        self.renderer = None
        self.text_cache: dict[tuple[str, pygame.Font, int], tuple[Texture, pygame.Rect]] = {}  # (text, font, color) -> {texture, rect}

    def enable_gpu(self, ren):
        global draw_line, draw_rect, fill_rect, draw_quad, fill_quad, draw_triangle, fill_triangle
        global fill_display, T, subsurface, imgload, write
        global flip_x, rotate_ip
        
        self.renderer = ren

        def draw_line(ren, color, p1, p2):
            ren.draw_color = color
            ren.draw_line(p1, p2)

        def draw_rect(ren, color, rect, *args):
            ren.draw_color = color
            ren.draw_rect(rect)

        def fill_rect(ren, color, rect):
            ren.draw_blend_mode = 1
            ren.draw_color = color
            ren.fill_rect(rect)
            ren.draw_blend_mode = 0

        def draw_quad(ren, color, p1, p2, p3, p4):
            ren.draw_color = color
            ren.draw_quad(p1, p2, p3, p4)

        def fill_quad(ren, color, p1, p2, p3, p4):
            ren.draw_color = color
            ren.fill_quad(p1, p2, p3, p4)

        def draw_triplangle(ren, color, p1, p2, p3):
            ren.draw_color = color
            ren.draw_triangle(p1, p2, p3)

        def fill_triangle(ren, color, p1, p2, p3):
            ren.draw_color = color
            ren.fill_triangle(p1, p2, p3)

        def fill_display(ren, color):
            ren.draw_color = color
            ren.clear()
        
        def build(self):
            return CImage(Texture.from_surface(_glob.renderer, self.surf))
    
        SurfaceBuilder.build = build

        def T(*args, **kwargs):
            return CImage(Texture.from_surface(_glob.renderer, *args, **kwargs))
    
        def subsurface(surf, *args, **kwargs):
            return T(surf.subsurace(*args, **kwargs))
    
        def imgload(*path, scale=1, frames=None):
            img = pygame.image.load(Path(*path))
            if frames is None:
                return T(pygame.transform.scale_by(img, scale))
            elif frames == 1:
                return [T(pygame.transform.scale_by(img, scale))]
            else:
                imgs = []
                w, h = img.width / frames, img.height
                for x in range(frames):
                    imgs.append(T(pygame.transform.scale_by(img.subsurface(x * w, 0, w, h), scale)))
                return imgs
    
        def write(surf, anchor, text, font, color, x, y, alpha=255, blit=True, border=None, special_flags=0, tex=False, ignore=True):
            try:
                text_tex, text_rect = _glob.text_cache[(text, font, color)]
                setattr(text_rect, anchor, (int(x), int(y)))
                if blit:
                    surf.blit(text_tex, text_rect, special_flags=special_flags)
                return text_tex, text_rect
            
            except KeyError:
                if ignore:
                    # return
                    pass
                if border is not None:
                    bc, bw = border, 1
                    write(surf, anchor, text, font, bc, x - bw, y - bw, special_flags=special_flags),
                    write(surf, anchor, text, font, bc, x + bw, y - bw, special_flags=special_flags),
                    write(surf, anchor, text, font, bc, x - bw, y + bw, special_flags=special_flags),
                    write(surf, anchor, text, font, bc, x + bw, y + bw, special_flags=special_flags)

                text_surf = font.render(str(text), True, color)
                if tex:
                    text_surf = Texture.from_surface(surf, text)
                    text_surf.alpha = alpha
                else:
                    text_surf.set_alpha(alpha)
                text_tex = T(text_surf)
                text_rect = text_surf.get_rect()

                setattr(text_rect, anchor, (int(x), int(y)))
                if blit:
                    surf.blit(text_tex, text_rect, special_flags=special_flags)

                return _glob.text_cache.setdefault((text, font, color), (text_tex, text_rect))
    
        def flip_x(tex):
            img = Image(tex)
            img.flip_x = True
            return img
    
        def rotate_ip(img: Image, angle):
            img.angle = angle

        self.hwaccel = True


_glob = _Global()


def set_pyengine_gpu(ren):
    _glob.enable_gpu(ren)


def warp(surf: pygame.Surface,
         warp_pts,
         smooth=True,
         out: pygame.Surface = None) -> typing.Tuple[pygame.Surface, pygame.Rect]:
    """Stretches a pygame surface to fill a quad using cv2's perspective warp.

        Args:
            surf: The surface to transform.
            warp_pts: A list of four xy coordinates representing the polygon to fill.
                Points should be specified in clockwise order starting from the top left.
            smooth: Whether to use linear interpolation for the image transformation.
                If false, nearest neighbor will be used.
            out: An optional surface to use for the final output. If None or not
                the correct size, a new surface will be made instead.

        Returns:
            [0]: A Surface containing the warped image.
            [1]: A Rect describing where to blit the output surface to make its coordinates
                match the input coordinates.
    """
    if len(warp_pts) != 4:
        raise ValueError("warp_pts must contain four points")

    w, h = surf.get_size()
    is_alpha = surf.get_flags() & pygame.SRCALPHA

    # XXX throughout this method we need to swap x and y coordinates
    # when we pass stuff between pygame and cv2. I'm not sure why .-.
    src_corners = numpy.float32([(0, 0), (0, w), (h, w), (h, 0)])
    quad = [tuple(reversed(p)) for p in warp_pts]

    # find the bounding box of warp points
    # (this gives the size and position of the final output surface).
    min_x, max_x = float('inf'), -float('inf')
    min_y, max_y = float('inf'), -float('inf')
    for p in quad:
        min_x, max_x = min(min_x, p[0]), max(max_x, p[0])
        min_y, max_y = min(min_y, p[1]), max(max_y, p[1])
    warp_bounding_box = pygame.Rect(int(min_x), int(min_y),
                                    int(max_x - min_x),
                                    int(max_y - min_y))

    shifted_quad = [(p[0] - min_x, p[1] - min_y) for p in quad]
    dst_corners = numpy.float32(shifted_quad)

    mat = cv2.getPerspectiveTransform(src_corners, dst_corners)

    orig_rgb = pygame.surfarray.pixels3d(surf)

    flags = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
    out_rgb = cv2.warpPerspective(orig_rgb, mat, warp_bounding_box.size, flags=flags)

    if out is None or out.get_size() != out_rgb.shape[0:2]:
        out = pygame.Surface(out_rgb.shape[0:2], pygame.SRCALPHA if is_alpha else 0)

    pygame.surfarray.blit_array(out, out_rgb)

    if is_alpha:
        orig_alpha = pygame.surfarray.pixels_alpha(surf)
        out_alpha = cv2.warpPerspective(orig_alpha, mat, warp_bounding_box.size, flags=flags)
        alpha_px = pygame.surfarray.pixels_alpha(out)
        alpha_px[:] = out_alpha
    else:
        out.set_colorkey(surf.get_colorkey())

    # XXX swap x and y once again...
    return out, pygame.Rect(warp_bounding_box.y, warp_bounding_box.x,
                            warp_bounding_box.h, warp_bounding_box.w)


def get_rotation_matrix_x(angle_x):
    rotation_x = array([[1, 0, 0],
                    [0, cos(angle_x), -sin(angle_x)],
                    [0, sin(angle_x), cos(angle_x)]])
    return rotation_x


def get_rotation_matrix_y(angle_y):
    rotation_y = array([[cos(angle_y), 0, sin(angle_y)],
                    [0, 1, 0],
                    [-sin(angle_y), 0, cos(angle_y)]])
    return rotation_y


def get_rotation_matrix_z(angle_z):
    rotation_z = array([[cos(angle_z), -sin(angle_z), 0],
                    [sin(angle_z), cos(angle_z), 0],
                    [0, 0, 1]])
    return rotation_z


def shoelace(xs, ys):
    left_shoelace = sum(xs[i] * ys[i + 1] for i in range(len(xs) - 1))
    right_shoelace = sum(ys[i] * xs[i + 1] for i in range(len(xs) - 1))
    signed_area = 0.5 * (left_shoelace - right_shoelace)
    return signed_area


def polygon_orientation(vertices):
    """
    Determine polygon orientation using:
    - Find vertex A with smallest y (and largest x if tie)
    - Let B = previous vertex, C = next vertex
    - Compute sign of cross product of vectors AB and AC
    
    vertices: list of (x, y) tuples, polygon vertices in order
    
    Returns:
    - 1 if counter-clockwise
    - -1 if clockwise
    - 0 if degenerate (colinear)
    """
    n = len(vertices)
    if n < 3:
        return 0  # Not a polygon
    
    # Find index of vertex with smallest y, breaking ties by largest x
    min_idx = 0
    for i in range(1, n):
        if (vertices[i][1] < vertices[min_idx][1]) or \
           (vertices[i][1] == vertices[min_idx][1] and vertices[i][0] > vertices[min_idx][0]):
            min_idx = i
    
    A = vertices[min_idx]
    B = vertices[(min_idx - 1) % n]  # Previous vertex (wrap around)
    C = vertices[(min_idx + 1) % n]  # Next vertex (wrap around)
    
    # Vector AB
    AB = (B[0] - A[0], B[1] - A[1])
    # Vector AC
    AC = (C[0] - A[0], C[1] - A[1])
    
    # Cross product AB x AC (2D)
    cross = AB[0] * AC[1] - AB[1] * AC[0]
    
    if cross > 0:
        return -1  # Counter-clockwise (WRONG)
    elif cross < 0:
        return 1  # Clockwise (CORRECT)
    else:
        return 0  # Colinear / degenerate (WRONG)
    

# surfaces
def circle(radius, color=BLACK):
    ret = pygame.Surface((radius * 2 + 1, radius * 2 + 1), pygame.SRCALPHA)
    #pygame.gfxdraw.filled_circle(ret, radius, radius, radius, color)
    pygame.draw.circle(ret, color, (radius, radius), radius)
    return ret


def triangle(height, color=BLACK):
    ret = pygame.Surface((height, height), pygame.SRCALPHA)
    w, h = ret.get_size()
    pygame.draw.polygon(ret, color, ((0, h), (w / 2, 0), (w, h)))
    return ret


def aaellipse(width, height, color=BLACK):
    ret = pygame.Surface((width + 1, height + 1), pygame.SRCALPHA)
    pygame.gfxdraw.aaellipse(ret, width // 2, height // 2, width // 2, height // 2, color)
    return ret


def rgb_mult(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def rgb_to_grayscale(color):
    """
    r, g, b, a = color
    r *= 0.3
    g *= 0.59
    b *= 0.11
    gray = r + g + b
    gray = [gray] * 3
    """
    # you COULD do that^, or you could just do this instead:
    color = color[:3]
    gray = [sum(color) / len(color)] * 3
    return gray


def rot_pivot(surface, angle, pivot, offset, rotate=False):
    """Rotate the surface around the pivot point.

    Args:
        surface (pygame.Surface): The surface that is to be rotated.
        angle (float): Rotate by this angle.
        pivot (tuple, list, pygame.math.Vector2): The pivot point.
        offset (pygame.math.Vector2): This vector is added to the pivot.
    """
    # Rotate the image.
    if rotate:
        rotated_image = pygame.transform.rotate(surface, -angle)
    else:
        rotated_image = pygame.transform.rotozoom(surface, -angle, 1)
    rotated_offset = offset.rotate(angle)  # Rotate the offset vector.
    # Add the offset vector to the center/pivot point to shift the rect.
    rect = rotated_image.get_rect(center=pivot + rotated_offset)
    return rotated_image, rect  # Return the rotated image and shifted rect.


def rot_center(img, angle, pos):
    rot_img = rotozoom(img, angle, 1)
    new_rect = rot_img.get_rect(center=pos)
    return rot_img, new_rect


def palettize_img(img, palette):
    ret_img = img.copy()
    palette_colors = [palette.get_at((x, 0)) for x in range(palette.width)]
    for y in range(img.height):
        for x in range(img.width):
            cur = img.get_at((x, y))
            closest: list[pygame.Color, int] = []
            for color in palette_colors:
                dist = color_diff(cur, color)
                if not closest or dist < closest[1]:
                    closest = [color, dist]
            ret_img.set_at((x, y), closest[0])
    return ret_img


def borderize(img, color, thickness=1):
    mask = pygame.mask.from_surface(img)
    mask_surf = mask.to_surface(setcolor=color)
    mask_surf.set_colorkey(BLACK)
    surf = pygame.Surface([s + thickness * 2 for s in mask_surf.get_size()], pygame.SRCALPHA)
    poss = [[c * thickness for c in p] for p in [[1, 0], [2, 1], [1, 2], [0, 1]]]
    for pos in poss:
        surf.blit(mask_surf, pos)
    surf.blit(img, (thickness, thickness))
    return surf


def rgb2hex(rgb):
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)


def hex2rgb(hex_):
    return tuple(int(hex_.removeprefix("#")[i:i + 2], 16) for i in (0, 2, 4))


def contrast_color(rgb):
    return tuple(255 - c for c in rgb)


def rand_rgba():
    return tuple(rand(0, 255) for _ in range(3)) + (255,)


def grayscale(img):
    arr = pygame.surfarray.array3d(img)
    avgs = [[(r * 0.298 + g * 0.587 + b * 0.114) for (r, g, b) in col] for col in arr]
    arr = array([[(avg, avg, avg) for avg in col] for col in avgs])
    surf = pygame.surfarray.make_surface(arr)
    return surf


def center_window():
    os.environ["SDL_VIDEO_CENTERED"] = "1"


def imgload(*path, scale=1, frames=None):
    img = pygame.image.load(Path(*path))
    if frames is None:
        return pygame.transform.scale_by(img, scale)
    elif frames == 1:
        return [pygame.transform.scale_by(img, scale)]
    else:
        imgs = []
        w, h = img.width / frames, img.height
        for x in range(frames):
            imgs.append(pygame.transform.scale_by(img.subsurface(x * w, 0, w, h), scale))
        return imgs


def subsurface(surf, *args, **kwargs):
    return surf.subsurface(*args, **kwargs)


def write(surf, anchor, text, font, color, x, y, alpha=255, blit=True, border=None, special_flags=0, tex=False, ignore=True):
    try:
        return _glob.text_cache[(text, font, color)]
    except KeyError:
        if ignore:
            # return
            pass
        if border is not None:
            bc, bw = border, 1
            write(surf, anchor, text, font, bc, x - bw, y - bw, special_flags=special_flags),
            write(surf, anchor, text, font, bc, x + bw, y - bw, special_flags=special_flags),
            write(surf, anchor, text, font, bc, x - bw, y + bw, special_flags=special_flags),
            write(surf, anchor, text, font, bc, x + bw, y + bw, special_flags=special_flags)
        text = font.render(str(text), True, color)
        if tex:
            text = Texture.from_surface(surf, text)
            text.alpha = alpha
        else:
            text.set_alpha(alpha)
        text_rect = text.get_rect()
        setattr(text_rect, anchor, (int(x), int(y)))
        if blit:
            surf.blit(text, text_rect, special_flags=special_flags)
        return _glob.text_cache.setdefault((text, font, color), (text, text_rect))


def pg_to_pil(pg_img):
    return PIL.Image.frombytes("RGBA", pg_img.get_size(), pygame.image.tobytes(pg_img, "RGBA"))


def get_icon(type_, size=(34, 34)):
    w, h = size
    t = pygame.Surface((w, h), pygame.SRCALPHA)
    if type_ == "settings":
        c = w / 2
        of = w // 5
        pygame.gfxdraw.filled_circle(t, w // 2, h // 2, w // 4, DARK_GRAY)
        pygame.gfxdraw.filled_circle(t, w // 2, h // 2, w // 5, LIGHT_GRAY)
        pygame.gfxdraw.filled_circle(t, w // 2, h // 2, w // 8, DARK_GRAY)
        pygame.draw.rect(t, DARK_GRAY, (c - of / 2, c - of * 2, of, of))
        pygame.draw.rect(t, DARK_GRAY, (c - of / 2, c + of, of, of))
        pygame.draw.rect(t, DARK_GRAY, (c - of * 2, c - of / 2, of, of))
        pygame.draw.rect(t, DARK_GRAY, (c + of, c - of / 2, of, of))
    elif type_ == "cursor":
        pygame.draw.polygon(t, WHITE, ((13, 10), (19, 16), (19, 21)))
        pygame.draw.polygon(t, RED, ((13, 10), (19, 16), (24, 15)))
    elif type_ == "arrow":
        t = pygame.Surface((62, 34), pygame.SRCALPHA)
        pygame.draw.rect(t, WHITE, (0, 8, 34, 18))
        pygame.draw.polygon(t, WHITE, ((34, 0), (34, 34), (62, 17)))
    elif type_ == "option menu":
        pygame.draw.polygon(t, BLACK, ((0, 0), (8, 0), (4, 8)))
    elif type_ == "check":
        pygame.draw.aalines(t, BLACK, False, ((0, h / 5 * 3), (w / 5 * 2, h), (w, 0)))
    elif type_ == "grass":
        w, h = 5, 13
        t = pygame.Surface((w, h), pygame.SRCALPHA)
        c = rgb_mult((0, 128, 0), randf(0.5, 1.2))
        h = 12
        for x in range(w):
            t.fill(c, (x, 0, 1, nordis(h - rand(1, 7), 2)))
        t = pygame.transform.flip(t, False, True)
    return t


def swap_palette(surf, old_color, new_color):
    # TODO: perhaps rewrite with transparency
    old_color = old_color[:3]
    new = surf.copy()
    for y in range(surf.get_height()):
        for x in range(surf.get_width()):
            if new.get_at((x, y))[:3] == old_color:
                new.set_at((x, y), new_color)
    return new


def hide_cursor():
    #3pygame.mouse.set_cursor((w, h), (0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0))
    pygame.mouse.set_visible(False)


def average_z(vertices):
    return sum(v[2] for v in vertices) / len(vertices)


def get_binary_cursor(string, hotspot=(0, 0), black="x", white="-", xor=".", img=None):
    if img is None:
        size = (len(string[0]), len(string))
        xorm, andm = pygame.cursors.compile(string, black, white, xor)
        cursor = pygame.cursors.Cursor(size, hotspot, xorm, andm)
    else:
        return NotImplemented
    return cursor


# decorator functions
def loading(func, window, font, exit_command=None):
    class T:
        def __init__(self):
            self.t = None
    to = T()
    def load():
        while to.t.is_alive():
            write(window, "center", "Loading...", font, BLACK, *[s / 2 for s in window.get_size()])
        write(window, "center", "Finished!", font, BLACK, *[s / 2 for s in window.get_size()])
        if exit_command is not None:
            exit_command()
    def wrapper(*args, **kwargs):
        to.t = start_thread(func, args=args, kwargs=kwargs)
        start_thread(load)
    return wrapper


# classes
class Lerper:
    def __init__(self, speed, **kwargs):
        self.speed = speed
        self.lerps = []
        for k, v in kwargs.items():
            if k.startswith("target_"):
                setattr(self, k, v)
                setattr(self, k.removeprefix("target_"), v)
                self.lerps.append(k.removeprefix("target_"))
    
    def update(self):
        for cur in self.lerps:
            cur_value = getattr(self, cur)
            targ = "target_" + cur
            targ_value = getattr(self, targ)
            setattr(self, cur, cur_value + (targ_value - cur_value) * self.speed)

    def change_lerp(self, attr, value):
        setattr(self, attr, value)


class FillOptions(Enum):
    DELAUNAY = 0


class Crystal(Lerper):
    def __init__(self, renderer, vertices, point_colors, connections, fills, origin, mult, radius, xa=0, ya=0, za=0, xav=0, yav=0, zav=0, rotate=True, normals=False, normalize=False, mtl_file=None, backface_culling=True, speed=None, hwaccel=False, **kwargs):
        super().__init__(speed, **kwargs)
        self.hwaccel = hwaccel
        self.renderer = renderer
        self.normalize = normalize
        if mtl_file is not None:
            self.load_mtl(mtl_file)
        if isinstance(vertices, list):
            self.vertices = array(vertices)
            self.point_colors = point_colors
            self.fills = fills
        else:
            self.get_vertices_from_obj(vertices)
        if self.fills == FillOptions.DELAUNAY:
            self.get_delaunay()
        self.normals = normals
        self.r = radius
        self.circle_textures = [Texture.from_surface(self.renderer, circle(self.r, color)) for color in self.point_colors]
        self.connections = connections
        self.oox, self.ooy = origin
        self.m = mult
        self.xa, self.ya, self.za = xa, ya, za
        self.xav, self.yav, self.zav = xav, yav, zav
        self.rotate = rotate
        self.backface_culling = backface_culling
        self.update_lerp = True
        self.update_steps = 0
    
    @property
    def num_vertices(self):
        return len(self.vertices)

    # crystal update
    def update(self):
        self.calculations()
        self.draw()
    
    def load_mtl(self, mtl_file):
        self.mtl_file = mtl_file
        self.materials = []
        self.mtl_data = {}
        with open(self.mtl_file, "r") as f:
            for line in f.read().splitlines():
                args = line.split(" ")
                kw = args.pop(0)

                if kw == "newmtl":
                    current_mtl = args[0]
                    self.mtl_data[current_mtl] = {}

                elif kw == "Kd":
                    r, g, b = args
                    r, g, b = float(r) * 255, float(g) * 255, float(b) * 255
                    self.mtl_data[current_mtl]["Kd"] = (r, g, b)

    def calculations(self):
        # if self.update_steps > 0:
        #     return
        # self.update_steps += 1
        
        # lerper update
        if self.update_lerp:
            super().update()
        #
        self.points = []
        self.circles = []
        self.updated_vertices = []
        self.updated_normals = []
        # rotate
        if self.rotate:
            self.xa += self.xav
            self.ya += self.yav
            self.za += self.zav

        xa, ya, za = self.xa, self.ya, self.za
        for index, vertex in enumerate(self.vertices):
            # rotate the matrices
            vertex = vertex.dot(get_rotation_matrix_x(xa))
            vertex = vertex.dot(get_rotation_matrix_y(ya))
            vertex = vertex.dot(get_rotation_matrix_z(za))
            self.updated_vertices.append(vertex)
            # project the matrices
            pos = vertex.dot(orthogonal_projection_matrix)
            x, y = self.m * pos[0] + self.oox, self.m * pos[1] + self.ooy
            rect = pygame.Rect(x - self.r, y - self.r, self.r * 2, self.r * 2)
            self.points.append((x, y))
            self.circles.append([index, rect])
            
        if self.normals:
            for index, vector in enumerate(self.vertex_normals):
                # rotate the matrices
                vector = vector.dot(get_rotation_matrix_x(xa))
                vector = vector.dot(get_rotation_matrix_y(ya))
                vector_normal = vector.dot(get_rotation_matrix_z(za))
                self.updated_normals.append(vector_normal)

        # fills
        self.fill_vertices = [[fill[0], [self.updated_vertices[x] for x in fill[1]]] for fill in self.fills]
        self.fill_vertices = sorted(self.fill_vertices, key=lambda v: average_z(v[1]))
        self.fill_data = []

        for index, capsule in enumerate(self.fill_vertices):
            # init lel
            data = capsule[0]
            d = [data]
            vertices = capsule[1]
            if self.backface_culling and False:
                cond = polygon_orientation(vertices)
            else:
                cond = True
            if cond:
                # final poly projection calculation
                for vertex in vertices:
                    # project the matrices
                    pos = vertex.dot(orthogonal_projection_matrix)
                    x, y = self.m * pos[0] + self.oox, self.m * pos[1] + self.ooy
                    rect = pygame.Rect(x - self.r, y - self.r, self.r * 2, self.r * 2)
                    # self.renderer.blit(self.default_circle, rect)
                    d.append([x, y])
                self.fill_data.append(d)

    # crystal draw
    def draw(self):
        for index, data in enumerate(self.fill_data):
            self.fill_points(*data)
                
        for connection in self.connections:
            self.connect_points(*connection)

        for circle in self.circles:
            with suppress(IndexError):
                self.draw_circle(*circle)

    def save_to_file(self, path_):
        with open(path_, "w") as f:
            for vertex in self.vertices:
                str_vertex = f"v {' '.join(str(x) for x in vertex)}\n"
                f.write(str_vertex)
            f.write("\n")
            for fill in self.fills:
                str_fill = f"f {' '.join(str(x + 1) + '/1/1' for x in fill[1:])}\n"
                f.write(str_fill)

    def draw_circle(self, i, rect):
        self.renderer.blit(self.circle_textures[i], rect)

    def connect_points(self, line_color, *points, index=True):
        for i in range(len(points)):
            j = points[(i + 1) if i < len(points) - 1 else 0]
            i = points[i]
            if index:
                i, j = self.points[i], self.points[j]
            pygame.draw.line(self.renderer, line_color, i, j, 1)

    def fill_points(self, data, *points):
        # setup
        fill_color = data[0] if 0 < len(data) else False
        outline_color = data[1] if 1 < len(data) else False
        if isinstance(data[0], pygame.Surface):
            surf = data[0]
            fill_color = False
        else:
            surf = False
        # normals
        if self.normals:
            normal_index = data[2]
            normal = self.updated_normals[normal_index]
            vec = pygame.math.Vector3(list(normal))
            camera = pygame.math.Vector3(0, 0, 1)
            dot = vec.dot(camera)
            if dot < 0:
                return
            light_value = (dot + 1) / 2
            fill_color = rgb_mult(fill_color, light_value)
        # filling
        if surf:
            surf, rect = warp(surf, points)
            if _glob.hwaccel:
                surf = Texture.from_surface(self.renderer, surf)
            self.renderer.blit(surf, rect)
        if len(points) == 3:
            if fill_color:
                fill_triangle(self.renderer, fill_color, *points)
            if outline_color:
                draw_triangle(self.renderer, outline_color, *points)
        elif len(points) == 4:
            if fill_color:
                fill_quad(self.renderer, fill_color, *points)
            if outline_color:
                draw_quad(self.renderer, outline_color, *points)
        else:
            return

        # draw the normals (debug)
        if self.normals:
            farther = normal * 1.4
            pos = normal.dot(orthogonal_projection_matrix)
            x, y = 200 * pos[0] + self.oox, 200 * pos[1] + self.ooy
            orect = pygame.Rect(x - self.r, y - self.r, self.r * 3, self.r * 3)
            pos = farther.dot(orthogonal_projection_matrix)
            x, y = 200 * pos[0] + self.oox, 200 * pos[1] + self.ooy
            rect = pygame.Rect(x - self.r, y - self.r, self.r * 3, self.r * 3)
            # draw_line(self.renderer, fill_color, orect.topleft, rect.topleft)

    def map_texture(self, data, *points):
        surf = data[0]
        surf, rect = warp(surf, points)
        tex = Texture.from_surface(self.renderer, surf)
        self.renderer.blit(tex, rect)

    def get_vertices_from_obj(self, p):
        self.vertices = []
        self.fills = []
        self.vertex_normals = []
        self.updated_normals = []
        min_ = max_ = 0

        current_mtl = None

        with open(p) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue

                line = re.sub(r"\s*#.*$", "", line)
                split = line.split(" ")
                i = split[0]

                # vector
                if i == "v":
                    vertex = [float(x) for x in split[1:] if x]
                    for x in vertex:
                        if x > max_:
                            max_ = x
                        elif x < min_:
                            min_ = x
                        if abs(x) > max_:
                            max_ = x
                    self.vertices.append(vertex)

                # face
                elif i == "f":
                    face = []
                    datae = [x for x in split[1:] if x]
                    face.append([])
                    for data in datae:
                        if "//" in data:
                            vertex, normal = [int(x) - 1 for x in data.split("//")]
                        else:
                            vertex, uv, normal = [int(x) - 1 for x in data.split("/")]
                        face[-1].append(vertex)
                    if current_mtl is None:
                        face.insert(0, [[rand(0, 255) for _ in range(3)] + [255], False, normal])
                    else:
                        face.insert(0, [rgb_mult(self.mtl_data[current_mtl]["Kd"], randf(1, 1)), False, normal])
                    if len(face) <= 5:  # why?
                        self.fills.append(face)
                    
                    # if current_mtl is not None:
                    #     self.materials.append(current_mtl.rstrip("\n"))

                # vector normals
                elif i == "vn":
                    normal_coords = [float(x) for x in split[1:]]
                    self.vertex_normals.append(normal_coords)

                elif i == "usemtl":
                    current_mtl = split[1].rstrip("\n")

        self.vertex_normals = array(self.vertex_normals)
        # self.point_colors = [(255, 0, 0, 255)] * len(self.vertices)
        self.point_colors = []

        if self.normalize:
            self.vertices = [[x / max(abs(min_), abs(max_)) for x in vertex] for vertex in self.vertices]
        
        self.vertices = array(self.vertices)

    def get_delaunay(self):
        self.fills = []
        for poly in Delaunay(self.vertices).simplices:
            self.fills.append([[rgb_mult(DARK_GRAY, randf(0.8, 1.2)), WHITE], poly])


class PhysicsEntity:
    def __init__(self, win, size, space, x, y, m=5, r=5, d=1, e=1, w=None, h=None, body_type=pymunk.Body.DYNAMIC, shape_type="circle", color=RED, to_tex=False):
        self.win = win
        self.width, self.height = size
        self.space = space
        self.x, self.y = x, y
        self.color = color
        self.w, self.h = w, h
        self.body = pymunk.Body(body_type=body_type, mass=m)
        self.body.position = (x, self.height - y - (self.h if self.h is not None else 0))
        self.r = r
        self.shape_type = shape_type
        if shape_type == "circle":
            self.shape = pymunk.Circle(self.body, r)
            self.img = pygame.Surface([self.r * 2 + 1] * 2, pygame.SRCALPHA)
            pygame.gfxdraw.aacircle(self.img, self.r, self.r, self.r, self.color)
        elif shape_type == "rect":
            self.shape = pymunk.Poly(self.body, [
                (0, 0),
                (self.width, 0),
                (self.width, self.h),
                (0, self.h)
            ])
            self.img = pygame.Surface((w, h))
            self.img.fill(self.color)
        self.shape.density = d
        self.shape.elasticity = e
        self.space.add(self.body, self.shape)
        if to_tex:
            self.img = Texture.from_surface(self.win, self.img)

    def draw(self):
        if self.shape_type == "circle":
            body_pos = [self.body.position[0], self.height - self.body.position[1]]
            body_pos = [p - self.r for p in body_pos]
            self.body_rect = pygame.Rect(*body_pos, self.r * 2, self.r * 2)
        elif self.shape_type == "rect":
            body_pos = [self.body.position[0], self.height - self.body.position[1] - self.h]
            self.body_rect = pygame.Rect(body_pos, (self.w, self.h))
        self.win.blit(self.img, self.body_rect)

    def go(self, pos):
        self.body.position = (pos[0], self.height - pos[1])


class PhysicsEntityConnector:
    def __init__(self, win, size, space, src, dest, anchor_a=(0, 0), anchor_b=(0, 0), to_tex=False):
        self.win = win
        self.width, self.height = size
        self.space = space
        if src.body.body_type == STATIC and dest.body.body_type == STATIC:
            raise ValueError(f"One of the two physics entities must be a dynamic body instead of static ({src.body.body_type}, {dest.body.body_type})")
        self.src = src
        self.dest = dest
        self.anchor_a = anchor_a
        self.anchor_b = anchor_b
        self.joint = pymunk.PinJoint(self.src.body, self.dest.body, anchor_a=self.anchor_a, anchor_b=self.anchor_b)
        self.joint.collide_bodies = False
        self.space.add(self.joint)
        self.to_tex = to_tex

    def draw(self):
        src_pos = [self.src.body.position[0] + self.anchor_a[0], self.height - self.src.body.position[1] + self.anchor_a[1]]
        dest_pos = [self.dest.body.position[0] + self.anchor_b[0], self.height - self.dest.body.position[1] + self.anchor_b[1]]
        if self.to_tex:
            self.win.draw_color = (27, 120, 60, 255)
            self.win.draw_line(src_pos, dest_pos)
        else:
            pygame.draw.line(self.win, (27, 120, 60, 255), src_pos, dest_pos)


class _Key:
    def __repr__(self):
        return repr(self.codes)


    def __eq__(self, other):
        return other in self.codes


class _Enter(_Key):
    def __init__(self):
        self.codes = (pygame.K_RETURN, pygame.K_KP_ENTER)


class _Shift(_Key):
    def __init__(self):
        self.codes = (1, 8192, 1073742049)


class _Control(_Key):
    def __init__(self):
        self.codes = (64, 8256, 1073742048)


class _Option(_Key):
    def __init__(self):
        self.codes = (256, 512, 1073742050, 1073742054)


class _Command(_Key):
    def __init__(self):
        self.codes = (1024, 2048, 1073742051, 1073742055)


class _OsC(_Key):
    def __init__(self):
        if Platform.os in ("windows", "linux"):
            self.codes = K_CONTROL.codes
        elif Platform.os == "darwin":
            self.codes = K_COMMAND.codes


K_ENTER = _Enter()
K_SHIFT = _Shift()
K_CONTROL = _Control()
K_OPTION = _Option()
K_COMMAND = _Command()
K_OSC = _OsC()


# CPU rendering
def T(x):
    return x


class SmartSurface(pygame.Surface):
    writed = 0
    def __init__(self, *args, **kwargs):
        og_args = list(args)
        args = og_args[:]
        with suppress(ValueError):
            args.remove("notalpha")
        args = tuple(args)
        super().__init__(*args)
        if "notalpha" not in og_args:
            self = self.convert_alpha()

    def __repr__(self):
        return f"{type(self).__name__}(size={self.get_size()}, flags={hex(self.get_flags())})"

    def __reduce__(self):
        return (str, (self._tobytes,))

    def __deepcopy__(self, memo):
        return self._tobytes

    @property
    def _tobytes(self, mode="RGBA"):
        return pygame.image.tobytes(self, mode)

    @classmethod
    def from_surface(cls, surface):
        ret = cls(surface.get_size(), pygame.SRCALPHA).convert_alpha()
        ret.blit(surface, (0, 0))
        return ret

    @classmethod
    def from_string(cls, string, size, format="RGBA"):
        return cls.from_surface(pygame.image.frombytes(string, size, format))

    def cblit(self, surf, pos, anchor="center"):
        rect = surf.get_rect()
        setattr(rect, anchor, pos if not isinstance(pos, pygame.Rect) else pos.topleft)
        self.blit(surf, rect)

    def to_pil(self):
        return pg_to_pil(self)


class Font:
    def __init__(self, *p, char_list="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", spacing_color=RED):
        self.sprs = imgload(*p)
        self.char_list = char_list
        self.char_height = self.sprs.get_height()
        self.chars = {}
        self.spacing_color = spacing_color
        index = 0
        left = 0
        right = 0
        rgb = self.sprs.get_at((right, 0))
        while right < self.sprs.get_width():
            rgb = self.sprs.get_at((right, 0))
            if rgb == self.spacing_color:
                self.chars[self.char_list[index]] = self.sprs.subsurface(left, 0, right - left, self.char_height)
                right += 3
                left = right
                index += 1
            else:
                right += 1

    def render(self, surf, string, pos, anchor="center"):
        spacing = 3
        size_x = 0
        size_y = self.char_height
        str_widths = []
        for c in string:
            if c != " ":
                s = self.chars[c].get_width()
                size_x += s + spacing
                str_widths.append(s)
            else:
                size_x += 10
                str_widths.append(5)
        template = pygame.Surface((size_x, size_y), pygame.SRCALPHA)
        x = 0
        for i, c in enumerate(string):
            if c != " ":
                template.blit(self.chars[c], (x, 0))
            x += str_widths[i] + spacing
        template = template.subsurface(0, 0, template.get_width() - 3, template.get_height())
        rect = template.get_rect()
        setattr(rect, anchor, pos)
        surf.blit(template, rect)


class RadarChart:
    def __init__(self, win, data, x, y, r, line_color, font):
        self.win = win
        self.x, self.y = x, y
        self.r = r
        self.line_color = line_color
        self.max_ = len(data)
        self.max_value = max(data, key=lambda x: x[1])[1]
        self.angles = [a / self.max_ * 2 * pi + 1 / 12 * 2 * pi for a in range(self.max_)]
        self.texts = [d[0] for d in data]
        self.values = [d[1] for d in data]
        self.revals = self.values.copy()
        self.font = font

    def update(self):
        late_lines = []
        # update values
        for i, val in enumerate(self.values):
            self.values[i] += (self.revals[i] - val) * 0.08
        # render
        for i, angle in enumerate(self.angles):
            # var init
            num_r = 7
            sec_color = (120, 120, 120, 255)
            try:
                self.angles[i + 1]
            except IndexError:
                index_error = True
            else:
                index_error = False
            # borders
            for r in range(num_r + 1):
                color = self.line_color if r / num_r == 1 else sec_color
                if index_error:
                    nx, ny = r / num_r * self.r * cos(self.angles[0]), r / num_r * self.r * sin(self.angles[0])
                else:
                    nx, ny = r / num_r * self.r * cos(self.angles[i + 1]), r / num_r * self.r * sin(self.angles[i + 1])
                cx, cy = r / num_r * self.r * cos(angle), r / num_r * self.r * sin(angle)
                draw_line(self.win, color, (self.x + cx, self.y + cy), (self.x + nx, self.y + ny))
            draw_line(self.win, sec_color, (self.x + cx, self.y + cy), (self.x, self.y))
            # actul stat
            if index_error:
                nx, ny = self.values[0] / self.max_value * self.r * cos(self.angles[0]), self.values[0] / self.max_value * self.r * sin(self.angles[0])
            else:
                nx, ny = self.values[i + 1] / self.max_value * self.r * cos(self.angles[i + 1]), self.values[i + 1] / self.max_value * self.r * sin(self.angles[i + 1])
            cx, cy = self.values[i] / self.max_value * self.r * cos(self.angles[i]), self.values[i] / self.max_value * self.r * sin(self.angles[i])
            late_lines.append((SLIME_GREEN, (self.x + cx, self.y + cy), (self.x + nx, self.y + ny)))
            # text
            text = self.texts[i]
            tm = 1.2
            tx, ty = self.r * tm * cos(self.angles[i]), self.r * tm * sin(self.angles[i])
            write(self.win, "center", text, self.font, WHITE, self.x + tx, self.y + ty, tex=False)
        for data in late_lines:
            draw_line(self.win, *data)

    def revaluate(self, *pairs):
        for text, value in pairs:
            self.revals[self.texts.index(text)] = value
