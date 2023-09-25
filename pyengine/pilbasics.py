import PIL.Image
import PIL.ImageFilter
import PIL.ImageColor
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageOps
import pygame
from math import cos, pi, atan, atan2, degrees, radians


# shapes
...

# functions
def get_isometric(img):
    # img = img.resize((img.width, int(img.height * cos(degrees(30)))))
    skew = 0.5
    left_angle = degrees(atan2((img.width * (skew + 1) - img.width), img.height))
    left = pil_skew(img, -skew).rotate(-left_angle, PIL.Image.NEAREST, expand=True)
    left = left.crop(left.getbbox())
    right = PIL.ImageOps.mirror(left)
    top_angle = degrees(atan(skew))
    top = pil_skew(img, -skew).rotate(30, PIL.Image.NEAREST, expand=True)
    top = top.crop(top.getbbox())
    yo = top.height // 2
    w, h = left.width, yo + left.height
    ret = PIL.Image.new("RGBA", (w * 2, h))
    ret.paste(left, (0, yo))
    ret.paste(right, (w, yo))
    ret.paste(top, (0, 0), top)
    ret.show()


def pil_skew(img, m):
    width, height = img.size
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    img = img.transform((new_width, height), PIL.Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, 0, 1, 0), PIL.Image.BICUBIC)
    return img


def round_corners(pil_img, radius):
    circle = PIL.Image.new('L', (radius * 2, radius * 2), 0)
    draw = PIL.ImageDraw.Draw(circle)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)
    alpha = PIL.Image.new('L', pil_img.size, 255)
    w, h = pil_img.size
    alpha.paste(circle.crop((0, 0, radius, radius)), (0, 0))
    alpha.paste(circle.crop((0, radius, radius, radius * 2)), (0, h - radius))
    alpha.paste(circle.crop((radius, 0, radius * 2, radius)), (w - radius, 0))
    alpha.paste(circle.crop((radius, radius, radius * 2, radius * 2)), (w - radius, h - radius))
    pil_img.putalpha(alpha)
    return pil_img


def hex_to_rgb(value):
    return PIL.ImageColor.getcolor(value, "RGB")


def pil_rot(pil_img, angle):
    return pil_img.rotate(angle, PIL.Image.NEAREST, expand=1)


def pil_blur(pil_img, radius=2):
    return pil_img.filter(PIL.ImageFilter.GaussianBlur(radius=radius))


def pil_to_pg(pil_img):
    data = pil_img.tobytes()
    size = pil_img.size
    mode = pil_img.mode
    return pygame.image.fromstring(data, size, mode)


def pil_rect2pg(pil_rect):
    return (pil_rect[0], pil_rect[1], pil_rect[2] - pil_rect[0], pil_rect[3] - pil_rect[1])


def pil_pixelate(pil_img, size):
    small = pil_img.resize(size, resample=PIL.Image.BILINEAR)
    result = small.resize(pil_img.size, PIL.Image.NEAREST)
    return result


def pil_contrast(pil_img, factor=2):
    enhancer = PIL.ImageEnhance.Contrast(pil_img)
    ret = enhancer.enhance(factor)
    return ret
