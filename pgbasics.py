from .imports import *
from .basics import *
from .pilbasics import pil_to_pg
import pygame
from pygame.locals import *
import pygame.gfxdraw
from colorsys import rgb_to_hsv, hsv_to_rgb, rgb_to_hls, hls_to_rgb


pygame.init()

pgscale = pygame.transform.scale
ticks = pygame.time.get_ticks
scale = pygame.transform.scale
scale2x = pygame.transform.scale2x
rotate = pygame.transform.rotate
rotozoom = pygame.transform.rotozoom
flip = pygame.transform.flip
set_cursor = pygame.mouse.set_cursor

# event costants
is_left_click = lambda event: event.type == pygame.MOUSEBUTTONDOWN and event.button == 1

# colors
BLACK =         (  0,   0,   0)
WHITE =         (255, 255, 255)
LIGHT_GRAY =    (180, 180, 180)
GRAY =          (120, 120, 120)
DARK_GRAY =     ( 80,  80,  80)
RED =           (255,   0,   0)
DARK_RED =      ( 56,   0,   0)
LIGHT_PINK =    (255, 247, 247)
GREEN =         (  0, 150,   0)
MOSS_GREEN =    ( 98, 138,  56)
DARK_GREEN =    (  0,  70,   0)
DARKISH_GREEN = (  0,  120,  0)
LIGHT_GREEN =   (  0, 255,   0)
MINT =          (186, 227, 209)
YELLOW =        (255, 255,   0)
YELLOW_ORANGE = (255, 174,  66)
SKIN_COLOR =    (255, 219, 172)
GOLD_YELLOW =   (255, 214,   0)
POWDER_BLUE =   (176, 224, 230)
WATER_BLUE =    (  0, 191, 255)
SKY_BLUE =      (220, 248, 255)
LIGHT_BLUE =    (137, 209, 254)
SMOKE_BLUE =    (154, 203, 255)
DARK_BLUE =     (  0,   0,  50)
PURPLE =        (153,  50, 204)
ORANGE =        (255,  94,  19)
BROWN =         (125,  70,   0)
DARK_BROWN =    ( 87,  45,   7)
LIGHT_BROWN =   (149,  85,   0)
DARK_BROWN =    (101,  67,  33)
PINK =          (255, 192, 203)

INF = "\u221e"  # infinity symbol (unicode)


# surfaces
def circle(radius, color=BLACK):
    ret = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
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
def rgb2hex(rgb):
    return "#{0:02x}{1:02x}{2:02x}".format(*rgb)
    

def hex2rgb(hex_):
    return tuple(int(hex_.removeprefix("#")[i:i + 2], 16) for i in (0, 2, 4))
    
    
def contrast_color(rgb):
    return tuple(255 - c for c in rgb)


def depr_contrast_color(rgb):
    hsv = rgb_to_hsv(*rgb)
    return hsv_to_rgb((hsv[0] + 0.5) % 1, hsv[1], hsv[2])


def rot_center(img, angle, x, y):
    rot_img = rotozoom(img, angle, 1)
    new_rect = rot_img.get_rect(center=img.get_rect(center=(x, y)).center)
    return rot_img, new_rect
    
    
def crop_transparent(pg_img):
    pil_img = pg2pil(pg_img)
    pil_img = pil_img.crop(pil_img.getbbox())
    pg_img = pil2pg(pil_img)
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
    
    
def angle_to_vel(angle, speed):
    vx = cos(angle) * speed
    vy = sin(angle) * speed
    return vx, vy
    
    
def two_pos_to_vel(pos, mouse, speed=1):
    return angle_to_vel(two_pos_to_angle(pos, mouse), speed)


def bar_rgb():
    """ RED, ORANGE, YELLOW, GREEN """
    return (lerp(RED, ORANGE, 33) + lerp(ORANGE, YELLOW, 33) + lerp(YELLOW, LIGHT_GREEN, 33))
   

def rand_rgb():
    return tuple(rand(0, 255) for _ in range(3))


def distance(rect1, rect2):
    try:
        return hypot(abs(rect1.x - rect2.x), abs(rect1.y - rect2.y))
    except AttributeError:
        return hypot(abs(rect1[0] - rect2[0]), abs(rect1[1] - rect2[1]))


def center_window():
    os.environ["SDL_VIDEO_CENTERED"] = "1"


def scalex(img, ratio):
    return pygame.transform.scale(img, [int(s * ratio) for s in img.get_size()])


def shrink2x(img):
    return pygame.transform.scale(img, [s / 2 for s in img.get_size()])


def point_in_mask(point, mask, rect):
    if rect.collidepoint(point):
        pos_in_mask = (point[0] - rect.x, point[1] - rect.y)
        if mask.get_at(pos_in_mask):
            return True
        else:
            return False
    else:
        return False


def scale3x(img):
    return pygame.transform.scale(img, [s * 3 for s in img.get_size()])


def imgload(*path_, alpha=True, colorkey=None, frames=None, frame_pause=0, scale=1):
    if frames is None:
        ret = pygame.image.load(path(*path_))
    else:
        ret = []
        img = pygame.image.load(path(*path_))
        frames = (frames, img.get_width() / frames)
        for i in range(frames[0]):
            ret.append(img.subsurface(i * frames[1], 0, frames[1], img.get_height()))
        for i in range(frame_pause):
            ret.append(ret[0])
    if isinstance(ret, list):
        for i, r in enumerate(ret):
            ret[i] = scalex(r.convert_alpha() if alpha else r.convert(), scale)
    elif isinstance(ret, pygame.Surface):
        ret = scalex(ret.convert_alpha() if alpha else ret.convert(), scale)
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


def write(surf, anchor, text, font, color, x, y):
    text = font.render(str(text), True, color)
    text_rect = text.get_rect()
    setattr(text_rect, anchor, (int(x), int(y)))
    surf.blit(text, text_rect)


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
    return PIL.Image.frombytes("RGBA", pg_img.get_size(), pygame.image.tostring(pg_img, "RGBA"))


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


def img_mult(img, mult):
    """
    pil_img = pg_to_pil(img)
    enhancer = PIL.ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(0.5)
    data = pil_img.tobytes()
    size = pil_img.size
    mode = pil_img.mode
    return pygame.image.fromstring(data, size, mode).convert_alpha()
    """
    ret_img = img.copy()
    for y in range(img.get_height()):
        for x in range(img.get_width()):
            rgba = img.get_at((x, y))
            if ret_img.get_at((x, y)) != (0, 0, 0, 0):
                ret_img.set_at((x, y), rgb_mult(rgba[:3], mult))
    return ret_img


def swap_palette(surf, old_color, new_color):
    img = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    img.blit(surf, (0, 0))
    for y in range(img.get_height()):
        for x in range(img.get_width()):
            rgba = surf.get_at((x, y))
            color = rgba[:-1]
            if color == old_color:
                img.set_at((x, y), new_color)
    img = img.convert_alpha()
    return img


def hide_cursor(w, h):
    #3pygame.mouse.set_cursor((w, h), (0, 0), (0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0))
    pygame.cursor.set_visible(False)
    
    
# decorator functions
def depr_loading(func, window, font, exit_command=None):
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
class _Enter:
    def __eq__(self, other):
        return other in (pygame.K_RETURN, pygame.K_KP_ENTER)
        
        
class _Shift:
    def __eq__(self, other):
        return other in (1, 8192)
    
    
class _Control:
    def __eq__(self, other):
        return other in (64, 8256)
        
        
K_ENTER = _Enter()
K_SHIFT = _Shift()
K_CTRL = _Control()


class SmartSurface(pygame.Surface):
    def __repr__(self):
        return f"{type(self).__name__}(size={self.get_size()}, flags={hex(self.get_flags())})"
    
    def __reduce__(self):
        return (str, (self._tostring,))
    
    def __deepcopy__(self, memo):
        return self._tostring
    
    @property
    def _tostring(self, mode="RGBA"):
        return pygame.image.tostring(self, mode)
    
    @classmethod
    def from_surface(cls, surface):
        ret = cls(surface.get_size(), pygame.SRCALPHA)
        ret.blit(surface, (0, 0))
        return ret
        
    @classmethod
    def from_string(cls, string, size, format):
        return cls.from_surface(pygame.image.fromstring(string, size, format))
    
    def cblit(self, surf, pos, anchor="center"):
        rect = surf.get_rect()
        setattr(rect, anchor, pos if not isinstance(pos, pygame.Rect) else pos.topleft)
        self.blit(surf, rect)
    
    def to_pil(self):
        return pg2pil(self)
    

class SmartGroup(pygame.sprite.Group):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
    
    def function(self):
        for spr in self:
            spr.function()
     
    def index(self, val):
        return self.sprites().index(val)
    
    def function(self):
        for spr in self:
            try:
                spr.draw()
                spr.update()
            except AttributeError:
                raise AttributeError(f"All sprites in the group must have a \"draw()\" and \"update()\" method. Object {spr} did not have that/those method(s).")
                

class BaseSprite:
    def __init__(self, group):
        self.group = group

    def kill(self):
        self.group.sprites().remove(self)


class Font:
    def __init__(self, name, char_list="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", spacing_color=RED):
        self.sprs = pygame.image.load(path("Fonts", name + ".png"))
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
