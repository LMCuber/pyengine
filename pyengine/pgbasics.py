from .imports import *
from .basics import *
from .pilbasics import pil_to_pg
import pygame
from pygame.locals import *
from pygame._sdl2.video import Window, Renderer, Texture, Image
import pygame.gfxdraw
import pygame.midi
import pymunk
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_hls, hls_to_rgb
from numpy import array


pygame.init()
pygame.midi.init()

pgscale = pygame.transform.scale
ticks = pygame.time.get_ticks
scale = pygame.transform.scale
scale2x = pygame.transform.scale2x
rotate = pygame.transform.rotate
rotozoom = pygame.transform.rotozoom
flip = pygame.transform.flip
set_cursor = pygame.mouse.set_cursor

STATIC = pymunk.Body.STATIC
DYNAMIC = pymunk.Body.DYNAMIC
KINEMATIC = pymunk.Body.KINEMATIC

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


# functions
def draw_line(ren, color, p1, p2):
    ren.draw_color = color
    ren.draw_line(p1, p2)


def draw_rect(ren, color, rect):
    ren.draw_color = color
    ren.draw_rect(rect)


def fill_rect(ren, color, rect):
    ren.draw_color = color
    ren.fill_rect(rect)


def draw_quad(ren, color, p1, p2, p3, p4):
    ren.draw_color = color
    ren.draw_quad(p1, p2, p3, p4)


def fill_quad(ren, color, p1, p2, p3, p4):
    ren.draw_color = color
    ren.fill_quad(p1, p2, p3, p4)


def draw_triangle(ren, color, p1, p2, p3):
    ren.draw_color = color
    ren.draw_triangle(p1, p2, p3)


def fill_triangle(ren, color, p1, p2, p3):
    ren.draw_color = color
    ren.fill_triangle(p1, p2, p3)


def mult_matrix(a, b):
    a_rows = len(a)
    a_cols = len(a[0])

    b_rows = len(b)
    b_cols = len(b[0])
    # Dot product matrix dimentions = a_rows x b_cols
    product = [[0 for _ in range(b_cols)] for _ in range(a_rows)]

    if a_cols == b_rows:
        for i in range(a_rows):
            for j in range(b_cols):
                for k in range(b_rows):
                    product[i][j] += a[i][k] * b[k][j]
    else:
        print("INCOMPATIBLE MATRIX SIZES")
    return product


def rot_pivot(image, pos, originPos, angle):
    image_rect = image.get_rect(topleft = (pos[0] - originPos[0], pos[1]-originPos[1]))
    offset_center_to_pivot = pygame.math.Vector2(pos) - image_rect.center
    rotated_offset = offset_center_to_pivot.rotate(-angle)
    rotated_image_center = (pos[0] - rotated_offset.x, pos[1] - rotated_offset.y)
    rotated_image = pygame.transform.rotate(image, angle)
    rotated_image_rect = rotated_image.get_rect(center = rotated_image_center)
    return rotated_image, rotated_image_rect


def color_diff_euclid(c1, c2):
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dz = c2[2] - c1[2]
    dist = sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return dist


def palettize_image(image, palette):
    img = pygame.Surface(image.get_size(), pygame.SRCALPHA)
    palette = imgload("assets", "Images", "Palettes", "sunset.png")
    colors = [palette.get_at((x, 0)) for x in range(palette.get_width())]
    # colors = [(0, 0, 128, 255), (194, 178, 128, 255), (0, 0, 0, 255), (137, 207, 240, 255)]
    for y in range(img.get_height()):
        for x in range(img.get_width()):
            rgba = tuple(img.get_at((x, y)))
            rgb = rgba[:3]
            if rgb in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] and rgba[-1] == 0:
                img.set_at((x, y), (0, 0, 0, 0))
                continue
            if rgba == (0, 0, 0, 0):
                continue
            min_ = (float("inf"), None)
            for color in colors:
                color = color[:3]
                diff = color_diff_euclid(rgb, color)
                if diff < min_[0]:
                    min_ = (diff, color)
            img.set_at((x, y), min_[1])
    return img


def get_rect_anim(size, border_radius=10, border_width=2, color=BLACK, colorkey_color=RED):
    w, h = size
    br = border_radius
    bw = border_width
    surf = pygame.Surface(size, pygame.SRCALPHA).convert_alpha()
    pygame.draw.rect(surf, color, (0, 0, *size), bw, br, br, br, br)
    surf.fill(colorkey_color, (0, 20, *size))
    surf.set_colorkey(colorkey_color)
    return surf


def diamond_square(power, mult=1):
    def _randomness():
        return randf(-10, 20)

    def _seed():
        return rand(4, 5)

    size = 2 ** power
    cs = size
    # dot = "\u2022"
    dot = 0
    world = [[dot] * (size + 1) for _ in range(size + 1)]
    world[0][0] = _seed()
    world[0][-1] = _seed()
    world[-1][0] = _seed()
    world[-1][-1] = _seed()
    surf = pygame.Surface((size, size))
    nth = 0
    for i in range(int(log2(size))):
        num_squares = 2 ** nth
        for y in range(num_squares):
            for x in range(num_squares):
                # init
                x = int(x)
                cs = int(cs)
                # square step
                points = [world[y * cs][x * cs], world[y * cs][x * cs + cs], world[y * cs + cs][x * cs], world[y * cs + cs][x * cs + cs]]
                avg = sum(points) / len(points) + _randomness()
                world[x * cs + cs // 2][y * cs + cs // 2] = avg
                # diamond step
                points = [world[x * cs + cs // 2][y * cs + cs // 2], world[y * cs][x * cs], world[y * cs][x * cs + cs]]
                avg = sum(points) / len(points) + _randomness()
                world[y * cs][x * cs + cs // 2] = avg  # top
                points = [world[x * cs + cs // 2][y * cs + cs // 2], world[y * cs][x * cs + cs], world[y * cs + cs][x * cs + cs]]
                avg = sum(points) / len(points) + _randomness()
                world[y * cs + cs // 2][x * cs + cs] = avg  # right
                points = [world[x * cs + cs // 2][y * cs + cs // 2], world[y * cs + cs][x * cs + cs], world[y * cs + cs][x * cs]]
                avg = sum(points) / len(points) + _randomness()
                world[y * cs + cs][x * cs + cs // 2] = avg  # bottom
                points = [world[x * cs + cs // 2][y * cs + cs // 2], world[y * cs + cs][x * cs], world[y * cs][x * cs]]
                avg = sum(points) / len(points) + _randomness()
                world[y * cs + cs // 2][x * cs] = avg  # left
        # next iteration setup
        cs /= 2
        nth += 1
    world = [[round(x, 1) if x > 0 else 0 for x in y] for y in world]
    highest = max([x for y in world for x in y])
    rects = []
    for yi, y in enumerate(world):
        for xi, x in enumerate(y):
            base = 1
            color = [base * round(x / highest * 255 / base) + 50 for _ in range(3)]
            if max(color) > 255:
                color = [255, 255, 255]
            elif min(color) < 0:
                color = [0, 0, 0]
            surf.set_at((xi * mult, yi * mult), color)
            # rects.append(((xi * mult, yi * mult, mult, mult), [rand(0, 255)] * 3))
    return surf


def pmotion(og_img, sxvel, syvel, dx, sx, sy, gravity):
    # calculating the quadratic function
    ytop = -((syvel ** 2) / (2 * -gravity))
    tair = (syvel / (-gravity)) * 2
    dist = sxvel * tair
    xtop = dist / 2
    top = (xtop, ytop)
    try:
        a = -(ytop / xtop ** 2)
    except ZeroDivisionError:
        return [], [], og_img
    equ = f"f(x) = {a}(x - {xtop})^2 + {ytop}"
    f = lambda x: a * (x - xtop) ** 2 + ytop
    # calculating the derivative of f
    parab = []
    step = 3
    equ = f"f'(x) = {2 * a}x + {(2 * a * -xtop)}"
    fp = lambda x: (2 * a * x) + (2 * a * -xtop)
    for x in range(0, int(xtop * 2), step if xtop >= 0 else -step):
        parab.append((sx + x, sy - int(f(x))))
    # calculating the tangent line function
    x1, y1 = dx, int(f(dx))
    m = fp(dx)
    ft = lambda x: m * x + m * -x1 + y1
    tangent = ((sx, sy - ft(0)), (sx + xtop * 2, sy - ft(xtop * 2)))
    # changing the angle accordingly
    rc = m
    ang = degrees(atan2(m, 1))
    img = pygame.transform.rotozoom(og_img, ang, 1)
    img.set_colorkey(BLACK)
    return parab, tangent, img


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


def depr_contrast_color(rgb):
    hsv = rgb_to_hsv(*rgb)
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])


def rot_center(img, angle, pos):
    rot_img = rotozoom(img, angle, 1)
    new_rect = rot_img.get_rect(center=pos)
    return rot_img, new_rect


def crop_transparent(pg_img):
    pil_img = pg_to_pil(pg_img)
    pil_img = pil_img.crop(pil_img.getbbox())
    pg_img = pil_to_pg(pil_img)
    return pg_img


def real_colorkey(img, color):
    og_surf = img.copy()
    og_surf.set_colorkey(color)
    blit_surf = pygame.Surface(og_surf.get_size(), pygame.SRCALPHA)
    blit_surf.blit(og_surf, (0, 0))
    return blit_surf


def set_volume(amount):
    if get_volume() != amount:
        pygame.mixer.music.set_volume(amount)


def get_volume():
    return pygame.mixer.music.get_volume()


def two_pos_to_angle(pos, mouse):
    dy = mouse[1] - pos[1]
    dx = mouse[0] - pos[0]
    angle = math.atan2(dy, dx)
    return angle


def angle_to_vel(angle, speed=1):
    vx = cos(angle) * speed
    vy = sin(angle) * speed
    return vx, vy


def two_pos_to_vel(pos, mouse, speed=1):
    return angle_to_vel(two_pos_to_angle(pos, mouse), speed)


def rand_rgba():
    return tuple(rand(0, 255) for _ in range(3)) + (255,)


def darken(pg_img, factor=0.7):
    ret = pg_to_pil(pg_img)
    enhancer = PIL.ImageEnhance.Brightness(ret)
    ret = enhancer.enhance(factor)
    ret = pil_to_pg(ret)
    return ret


def grayscale(img):
    arr = pygame.surfarray.array3d(img)
    avgs = [[(r * 0.298 + g * 0.587 + b * 0.114) for (r, g, b) in col] for col in arr]
    arr = array([[(avg, avg, avg) for avg in col] for col in avgs])
    surf = pygame.surfarray.make_surface(arr)
    return surf


def distance(rect1, rect2):
    try:
        return hypot(abs(rect1.x - rect2.x), abs(rect1.y - rect2.y))
    except AttributeError:
        return hypot(abs(rect1[0] - rect2[0]), abs(rect1[1] - rect2[1]))


def centroid(vertices):
    return [sum([v[i] for v in vertices]) / len(vertices) for i in range(3)]


def average_z(vertices):
    return sum([v[2] for v in vertices]) / len(vertices)


def distance_ndim(vertex):
    return sqrt(sum([(vertex[i]) ** 2 for i in range(len(vertex))]))


def center_window():
    os.environ["SDL_VIDEO_CENTERED"] = "1"


def point_in_mask(point, mask, rect):
    if rect.collidepoint(point):
        pos_in_mask = (point[0] - rect.x, point[1] - rect.y)
        return mask.get_at(pos_in_mask)
    return False



def shrink2x(img):
    return pygame.transform.scale(img, [s / 2 for s in img.get_size()])


def scale3x(img):
    return pygame.transform.scale(img, [s * 3 for s in img.get_size()])


def shrink3x(img):
    return pygame.transform.scale(img, [s / 3 for s in img.get_size()])


def imgload(*path_, after_func="convert_alpha", colorkey=None, frames=None, whitespace=0, frame_pause=0, end_frame=None, scale=1, rotation=0):
    if frames is None:
        ret = pygame.image.load(path(*path_))
    else:
        ret = []
        img = pygame.image.load(path(*path_))
        frames = (frames, img.get_width() / frames)
        for i in range(frames[0]):
            ret.append(img.subsurface(i * frames[1], 0, frames[1] - whitespace, img.get_height()))
        for i in range(frame_pause):
            ret.append(ret[0])
        if end_frame is not None:
            ret.append(ret[end_frame])
    if isinstance(ret, list):
        for i, r in enumerate(ret):
            ret[i] = rotate(pygame.transform.scale_by(getattr(r, after_func)() if after_func is not None else r, scale), rotation)
    elif isinstance(ret, pygame.Surface):
        ret = rotate(pygame.transform.scale_by(getattr(ret, after_func)() if after_func is not None else ret, scale), rotation)
    return ret


def off_screen(obj, w, h):
    if obj.rect.left >= w or obj.rect.right <= 0 or obj.rect.top >= h or obj.rect.bottom <= 0:
        return True
    else:
        return False


def lerp(start, end, amount):
    s = pygame.Color(start)
    e = pygame.Color(end)
    li = []
    for i in range(amount):
        li.append(s.lerp(e, fromperc(i, 1, amount)))
    return li


def lerp_img(start, end, amount, height):
    colors = lerp(start, end, amount)
    surf = pygame.Surface((len(colors), height))
    for x, color in enumerate(colors):
        pygame.draw.line(surf, color, (x, 0), (x, height), 1)
    return surf


def write(surf, anchor, text, font, color, x, y, alpha=255, blit=True, border=None, special_flags=0, tex=False):
    if border is not None:
        bc, bw = border, 1
        write(surf, anchor, text, font, bc, x - bw, y - bw),
        write(surf, anchor, text, font, bc, x + bw, y - bw),
        write(surf, anchor, text, font, bc, x - bw, y + bw),
        write(surf, anchor, text, font, bc, x + bw, y + bw)
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
    return text, text_rect


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


def fade_out(spr, amount=1, after_func=None):
    def thread_():
        for i in reversed(range(255)):
            spr.image.set_alpha(1)
        if after_func:
            after_func()
    start_thread(thread_)


def invert_rgb(rgb):
    return [255 - rgb for color in rgb]


def pg_to_pil(pg_img):
    return PIL.Image.frombytes("RGBA", pg_img.get_size(), pygame.image.tobytes(pg_img, "RGBA"))


def pg_rect_to_pil(pg_rect):
    return (pg_rect[0], pg_rect[1], pg_rect[0] + pg_rect[2], pg_rect[1] + pg_rect[3])


def rgb_mult(color, factor):
    ret_list = [int(c * factor) for c in color]
    for index, ret in enumerate(ret_list):
        if ret > 255:
            ret_list[index] = 255
        elif ret < 0:
            ret_list[index] = 0
    return ret_list


def img_mult(img, mult, type_=1):
    # t = epoch()
    if type_ == 1:
        pil_img = pg_to_pil(img)
        enhancer = PIL.ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(0.5)
        data = pil_img.tostring()
        size = pil_img.size
        mode = pil_img.mode
        return pygame.image.frombytes(data, size, mode).convert_alpha()
    elif type_ == 2:
        ret_img = img.copy()
        for y in range(img.get_height()):
            for x in range(img.get_width()):
                rgba = img.get_at((x, y))
                if ret_img.get_at((x, y)) != (0, 0, 0, 0):
                    ret_img.set_at((x, y), rgb_mult(rgba[:3], mult))
        return ret_img


def red_filter(img):
    img = pg_to_pil(img)
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = img.getpixel((x, y))
            pixels[x, y] = (r, 0, 0, a)
    return pil_to_pg(img)


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


# decorator functions
def get_binary_cursor(string, hotspot=(0, 0), black="x", white="-", xor=".", img=None):
    if img is None:
        size = (len(string[0]), len(string))
        xorm, andm = pygame.cursors.compile(string, black, white, xor)
        cursor = pygame.cursors.Cursor(size, hotspot, xorm, andm)
    else:
        return NotImplemented
    return cursor


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


# post-function constants
bar_rgb = (lerp(RED, ORANGE, 34) + lerp(ORANGE, YELLOW, 33) + lerp(YELLOW, LIGHT_GREEN, 34))


# classes
class CursorTrail:
    def __init__(self, surf, depth):
        self.surf = surf
        self.depth = depth
        self.data = []
        self.w, self.h = 10, 10
        for i in range(depth):
            img = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
            pygame.gfxdraw.filled_polygon(img, ((self.w / 2, 0), (self.w, self.h / 2), (self.w / 2, self.w), (0, self.h / 2)), WHITE)
            mult = i / depth
            # img = pygame.transform.scale_by(img, (mult * self.w, mult * self.h))
            try:
                img = Texture.from_surface(self.surf, img)
            except pygame._sdl2.sdl2.error:
                pass
            else:
                self.data.append([img, 0, 0, img.get_rect()])

    def update(self):
        self.draw()

    def draw(self):
        mouse = pygame.mouse.get_pos()
        for index in range(len(self.data)):
            img, x, y, rect = self.data[index]
            self.data[index][1] += (mouse[0] - x) * index / self.depth
            self.data[index][2] += (mouse[1] - y) * index / self.depth
            rect.x = self.data[index][1]
            rect.y = self.data[index][2]
            self.surf.blit(img, rect)


class Crystal:
    def __init__(self, renderer, vertices, point_colors, connections, fills, origin, mult, radius, xa=0, ya=0, za=0, xav=0, yav=0, zav=0, rotate=True, fill_as_connections=False, normals=False, normalize=False, **kwargs):
        self.__dict__.update(kwargs)
        self.renderer = renderer
        self.normalize = normalize
        if isinstance(vertices, str):
            self.get_vertices_from_obj(vertices)
            self.normals = True
        else:
            self.vertices = array(vertices)
            self.point_colors = point_colors
            self.fills = fills
            self.normals = False
        self.r = radius
        self.default_circle = Texture.from_surface(self.renderer, circle(5, RED))
        self.circle_textures = [Texture.from_surface(self.renderer, circle(self.r, color)) for color in self.point_colors]
        self.connections = connections
        if False:
            self.connections = [[YELLOW] + f[1:] for f in fills]
        self.ox, self.oy = origin
        self.m = mult
        self.xa, self.ya, self.za = xa, ya, za
        self.xav, self.yav, self.zav = xav, yav, zav
        self.rotate = rotate
        self.fill_as_connections = fill_as_connections
        # self.width = max([x[0] for x in self.vertices]) - min([x[0] for x in self.vertices]) * self.m
        # self.height = max([x[1] for x in self.vertices]) - min([x[1] for x in self.vertices]) * self.m
        # self.depth = max([x[2] for x in self.vertices]) - min([x[2] for x in self.vertices]) * self.m

    def update(self):
        self.draw()

    def draw(self):
        self.points = []
        self.circles = []
        self.updated_vertices = []
        self.updated_normals = []
        # rotate
        if self.rotate:
            self.xa += self.xav
            self.ya += self.yav
            self.za += self.zav
        for index, vertex in enumerate(self.vertices):
            # rotate the matrices
            vertex = vertex.dot(get_rotation_matrix_x(self.xa))
            vertex = vertex.dot(get_rotation_matrix_y(self.ya))
            vertex = vertex.dot(get_rotation_matrix_z(self.za))
            self.updated_vertices.append(vertex)
            # project the matrices
            pos = vertex.dot(orthogonal_projection_matrix)
            x, y = self.m * pos[0] + self.ox, self.m * pos[1] + self.oy
            rect = pygame.Rect(x - self.r, y - self.r, self.r * 2, self.r * 2)
            self.points.append((x, y))
            self.circles.append([index, rect])
        if self.normals:
            for index, vector in enumerate(self.vertex_normals):
                # rotate the matrices
                vector = vector.dot(get_rotation_matrix_x(self.xa))
                vector = vector.dot(get_rotation_matrix_y(self.ya))
                vector_normal = vector.dot(get_rotation_matrix_z(self.za))
                self.updated_normals.append(vector_normal)

        self.fill_vertices = [[self.updated_vertices[x] if isinstance(x, int) else x for x in data] for data in self.fills]
        self.fill_vertices = sorted(self.fill_vertices, key=lambda x: average_z(x[1:]))
        self.fill_data = []
        for index, data in enumerate(self.fill_vertices):
            # init lel
            color = data[0]
            d = [color]
            for vertex in data[1:]:
                # project the matrices
                pos = vertex.dot(orthogonal_projection_matrix)
                x, y = self.m * pos[0] + self.ox, self.m * pos[1] + self.oy
                rect = pygame.Rect(x - self.r, y - self.r, self.r * 2, self.r * 2)
                # self.renderer.blit(self.default_circle, rect)
                d.append([x, y])
            self.fill_data.append(d)

        for data in self.fill_data:
            if self.fill_as_connections and False:
                self.connect_points(*data, index=False)
            else:
                self.fill_points(*data)
        for connection in self.connections:
            self.connect_points(*connection)
        for circle in self.circles:
            with suppress(IndexError):
                self.draw_circle(*circle)

    def draw_circle(self, i, rect):
        self.renderer.blit(self.circle_textures[i], rect)

    def connect_points(self, line_color, *points, index=True):
        self.renderer.draw_color = line_color
        for i in range(len(points)):
            j = points[(i + 1) if i < len(points) - 1 else 0]
            i = points[i]
            if index:
                i, j = self.points[i], self.points[j]
            self.renderer.draw_line(i, j)

    def fill_points(self, data, *points):
        # setup
        fill_color = data[0] if 0 < len(data) else False
        outline_color = data[1] if 1 < len(data) else False
        # normals
        if self.normals:
            normal_index = data[2] if 2 < len(data) else False
            normal = self.updated_normals[normal_index]
            vec = pygame.math.Vector3(list(normal))
            camera = pygame.math.Vector3(0, 0, 1)
            dot = vec.dot(camera)
            fill_color = [0] + [int((dot + 1) / 2 * 255)] + [0, 255]
        # filling
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

        # draw the normals (debug)
        if self.normals:
            farther = normal * 1.4
            pos = normal.dot(orthogonal_projection_matrix)
            x, y = 200 * pos[0] + self.ox, 200 * pos[1] + self.oy
            orect = pygame.Rect(x - self.r, y - self.r, self.r * 3, self.r * 3)
            pos = farther.dot(orthogonal_projection_matrix)
            x, y = 200 * pos[0] + self.ox, 200 * pos[1] + self.oy
            rect = pygame.Rect(x - self.r, y - self.r, self.r * 3, self.r * 3)
            self.renderer.draw_color = fill_color
            self.renderer.draw_line(orect.topleft, rect.topleft)

    def get_vertices_from_obj(self, p):
        self.vertices = []
        self.fills = []
        self.vertex_normals = []
        self.updated_normals = []
        min_ = max_ = 0
        with open(p) as f:
            for line in f.readlines():
                split = line.split(" ")
                i = split[0]
                # vector
                if i == "v":
                    vertex = [float(x) for x in split[1:] if x]
                    # length = sqrt(abs(vertex[0] ** 2) + abs(vertex[1] ** 2) + abs(vertex[2] ** 2))
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
                    for data in datae:
                        if "//" in data:
                            vertex, normal = [int(x) - 1 for x in data.split("//")]
                        else:
                            vertex, uv, normal = [int(x) - 1 for x in data.split("/")]
                        face.append(vertex)
                    face.insert(0, [[rand(80, 120)] * 3 + [255], False, normal])
                    if len(face) <= 5:
                        self.fills.append(face)
                # vector normals
                elif i == "vn":
                    normal_coords = [float(x) for x in split[1:]]
                    self.vertex_normals.append(normal_coords)
        self.vertex_normals = array(self.vertex_normals)
        # self.point_colors = [(255, 0, 0, 255)] * len(self.vertices)
        self.point_colors = []
        if self.normalize:
            # self.vertices = [[(x - min_) / (max_ - min_) for x in vertex] for vertex in self.vertices]
            self.vertices = [[x / max_ for x in vertex] for vertex in self.vertices]
        self.vertices = array(self.vertices)


class CImage(Image):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_rect = self.texture.get_rect

    @property
    def width(self):
        return self.texture.width

    @property
    def height(self):
        return self.texture.height


class SmartVector:
    @property
    def left(self):
        return self.x

    @left.setter
    def left(self, value):
        self.x = value

    @property
    def right(self):
        return self.x + self.width

    @right.setter
    def right(self, value):
        self.x = value - self.width

    @property
    def centerx(self):
        return self.x + self.width / 2

    @centerx.setter
    def centerx(self, value):
        self.x = value - self.width / 2

    @property
    def centery(self):
        return self.y + self.height / 2

    @centery.setter
    def centery(self, value):
        self.y = value - self.height / 2

    @property
    def center(self):
        return (self.centerx, self.centery)

    @center.setter
    def center(self, value):
        self.centerx = value[0]
        self.centery = value[1]

    @property
    def top(self):
        return self.y

    @top.setter
    def top(self, value):
        self.y = value

    @property
    def bottom(self):
        return self.y + self.height

    @bottom.setter
    def bottom(self, value):
        self.y = value - self.height


class PhysicsEntity:
    def __init__(self, win, size, space, x, y, r=5, d=1, e=1, body_type=pymunk.Body.DYNAMIC):
        self.win = win
        self.width, self.height = size
        self.space = space
        self.x, self.y = x, y
        self.body = pymunk.Body(body_type=body_type)
        self.body.position = (x, self.height - y)
        self.r = r
        self.shape = pymunk.Circle(self.body, r)
        self.shape.density = d
        self.shape.elasticity = e
        self.space.add(self.body, self.shape)
        self.img = pygame.Surface([self.r * 2 + 1] * 2, pygame.SRCALPHA)
        pygame.gfxdraw.aacircle(self.img, self.r, self.r, self.r, RED)
        self.img = Texture.from_surface(self.win, self.img)

    def draw(self):
        body_pos = [self.body.position[0], self.height - self.body.position[1]]
        body_pos = [int(p - self.r) for p in body_pos]
        body_rect = pygame.Rect(*body_pos, self.r * 2, self.r * 2)
        self.win.blit(self.img, body_rect)

    def go(self, pos):
        self.body.position = (pos[0], self.height - pos[1])


class PhysicsEntityConnector:
    def __init__(self, win, size, space, src, dest):
        self.win = win
        self.width, self.height = size
        self.space = space
        self.src = src
        self.dest = dest
        self.joint = pymunk.PinJoint(self.src.body, self.dest.body)
        # self.limit_joint = pymunk.SlideJoint()
        self.space.add(self.joint)

    def draw(self):
        src_pos = [self.src.body.position[0], self.height - self.src.body.position[1]]
        dest_pos = [self.dest.body.position[0], self.height - self.dest.body.position[1]]
        self.win.draw_color = (27, 120, 60, 255)
        self.win.draw_line(src_pos, dest_pos)


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


class SmartSurface(pygame.Surface):
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


class SmartMask:
    def __init__(self, surf):
        self.surf = surf
        self.mask = pygame.mask.from_surface(surf)
        self.overlap = self.mask.overlap

    def __repr__(self):
        return repr(self.mask)

    def __deepcopy__(self, memo):
        return pygame.image.tobytes(self.surf, "RGBA")

    def __reduce__(self):
        return (str, (pygame.image.tobytes(self.surf, "RGBA",)))


class SmartGroup(list):
    def _function(self, attr):
        def _func():
            for spr in self:
                getattr(spr, attr, lambda_none)()
        return _func

    def __getattr__(self, attr):
        try:
            return vars(self)[attr]
        except KeyError:
            return self._function(attr)


class BaseSprite:
    def __init__(self, group):
        self.group = group

    def kill(self):
        self.group.sprites().remove(self)


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
