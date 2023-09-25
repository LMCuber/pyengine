from .basics import *
from .pgbasics import *
from .pilbasics import *


_DEF_WIDGET_POS = (100, 100)
WOOSH_TIME = 0.001
_cfonts = []
for i in range(101):
    _cfonts.append(pygame.font.SysFont("calibri", i))
keyboard_map = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+"}


class _Widget:
    def __pre_init__(self, font, text=None):
        self.font = font if font is not None else _eng.def_fonts[20]
        if text is not None:
            self.text = format_text(text)

    def __init__(self, image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, child, add, special_flags, tooltip, appends, as_child, *args, **kwargs):
        self.width = width if width is not None else self.image.get_width()
        self.height = height if height is not None else self.image.get_height()
        self.tooltip = tooltip
        if self.tooltip is not None:
            self.tooltip_img = pygame.Surface([s + 5 for s in _eng.def_tooltip_fonts[10].size(self.tooltip)])
            self.tooltip_img.fill((70, 70, 70))
            write(self.tooltip_img, "center", self.tooltip, _eng.def_tooltip_fonts[10], WHITE, *[s / 2 for s in self.tooltip_img.get_size()])
        self.special_flags = special_flags if special_flags is not None else []
        self.replace_og(image)
        self.surf = surf
        self.friends = []
        self.og_pos = pos
        self.anchor = anchor
        self.og_anchor = self.anchor
        self.exit_command = exit_command
        if template is None:
            self.disabled = disabled
            self.disable_type = disable_type
        elif template == "menu widget":
            self.disabled = True
            self.disable_type = "disable"
        if friends is not None:
            for friend in friends:
                self.add_friend(friend)
        if add:
            _mod.widgets.append(self)
        self.child = child
        if "rounded" in self.special_flags:
            img = pil_to_pg(round_corners(pg_to_pil(self.image), 10))
            self.image = img
            self.replace_og(img)
        self.visible_when = visible_when
        self.zooming = False
        self.as_child = as_child
        if appends is not None:
            for append in appends:
                append.append(self)
        setattr(self.rect, anchor, pos)
        self.image = Texture.from_surface(self.surf, self.image)
        Thread(target=self.zoom, args=["in"]).start()
        self.startup_command()

    def init_size(self, width=None, height=None, text=None):
        w = width if width is not None else self.font.size(text)[0] + 5
        h = height if height is not None else self.font.size(text)[1] + 5
        self.image = pygame.Surface((w, h), pygame.SRCALPHA)

    def _exec_command(self, command, after=None, *args, **kwargs):
        if callable(command):
            command(*args, **kwargs)
        if after == "out destroy":
            Thread(target=self.zoom, args=[after]).start()
        elif after == "kill":
            self._kill()

    def set_pos(self, pos, anchor="center"):
        setattr(self.rect, anchor, pos)
        self.og_pos = pos
        self.og_anchor = anchor

    def replace_og(self, image):
        self.og_img = image.copy()
        self.og_size = self.og_img.get_size()

    def startup_command(self):
        if not self.disabled:
            for wsc in _eng.widget_startup_commands:
                if wsc[0] is self.child:
                    wsc[1]()
                    break

    def enable(self):
        if self not in _mod.widgets:
            _mod.widgets.append(self)
        Thread(target=self.zoom, args=["in"]).start()
        self.disabled = False
        self.startup_command()

    def disable(self):
        Thread(target=self.zoom, args=["out"]).start()

    def zoom(self, type_):
        if type_ == "in":
            self.zooming = True
            for i in range(20):
                if False:
                    size = [round(fromperc((i + 1) * 5, s)) for s in self.og_size]
                    self.image = pygame.transform.scale(self.og_img, size)
                    self.rect = self.image.get_rect()
                setattr(self.rect, self.og_anchor, self.og_pos)
                time.sleep(WOOSH_TIME)
            self.zooming = False
        elif "out" in type_:
            self.zooming = True
            for i in reversed(range(20)):
                if False:
                    size = [round(fromperc((i + 1) * 5, s)) for s in self.og_size]
                    self.image = pygame.transform.scale(self.og_img, size)
                    self.rect = self.image.get_rect()
                setattr(self.rect, self.og_anchor, self.og_pos)
                time.sleep(WOOSH_TIME)
            if type_ == "out":
                self._exec_command(self.exit_command)
                self.disabled = True
            elif type_ == "out destroy":
                self._exec_command(self.exit_command, "kill")
            self.zooming = False

    def draw(self):
        self.surf.blit(self.image, self.rect)
        if hasattr(self, "icon_img"):
            self.surf.blit(self.icon_img, self.icon_rect)

    def add_friend(self, fr):
        self.friends.append(fr)
        fr.friends.append(self)
        self.friends = list(set(self.friends))
        fr.friends = list(set(fr.friends))

    def _kill(self):
        with suppress(ValueError):
            _mod.widgets.remove(self)

    def destroy(self):
        Thread(target=self.zoom, args=["out destroy"]).start()
        self._kill()


class ButtonBehavior:
    def update(self):
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            self.image.alpha = 150
        else:
            self.image.alpha = 255

    def click(self, left=True):
        if left:
            self.command(*[self] if self.pass_self else [])
        else:
            self.right_command(*[self] if self.pass_self_right else [])
        if self.click_effect:
            Thread(target=self.zoom, args=["out"]).start()


class _Overwriteable:
    def overwrite(self, text):
        w, h = [3 + size + 3 for size in self.font.size(str(text))]
        self.init_size(w, h, text)
        if isinstance(self, Checkbox):
            w += self.box.get_width() + 10
        self.image.fill(self.bg_color)
        if isinstance(self, Checkbox):
            write(self.image, "topleft", text, self.font, self.text_color, 30, 5)
        else:
            write(self.image, "center", text, self.font, self.text_color, *[s / 2 for s in self.image.get_size()])
        self.image = Texture.from_surface(self.surf, self.image)
        # self.replace_og(self.image)
        self.rect = self.image.get_rect(center=self.rect.center)

    def write_text(self, text, font, text_color, text_orien):
        if text_orien == "center":
            write(self.image, "center", text, font, text_color, *[s / 2 for s in self.image.get_size()])
        elif text_orien == "left":
            write(self.image, "midleft", text, font, text_color, 5, self.image.get_height() / 2)


class Button(_Widget, _Overwriteable, ButtonBehavior):
    def __init__(self, surf, text, command, pass_self=False, right_command=None, pass_self_right=False, width=None, height=None, pos=_DEF_WIDGET_POS, text_color=BLACK, text_orien="center", bg_color=WIDGET_GRAY, anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, click_effect=False, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font, text)
        self.text = text
        self.bg_color = bg_color
        self.click_effect = click_effect
        self.init_size(width, height, text)
        self.image.fill(bg_color)
        self.write_text(self.text, self.font, text_color, text_orien)
        self.rect = self.image.get_rect()
        self.command = command
        self.pass_self = pass_self
        self.right_command = right_command
        self.pass_self_right = pass_self_right
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def process_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if self.rect.collidepoint(pygame.mouse.get_pos()):
                    self.click()

            elif event.button == 3:
                if self.right_command is not None:
                    if self.rect.collidepoint(pygame.mouse.get_pos()):
                        self.click(left=False)


class ComboBox(_Widget, _Overwriteable, ButtonBehavior):
    def __init__(self, surf, text, combos, command=lambda_none, width=None, height=None, pos=_DEF_WIDGET_POS, text_color=BLACK, bg_color=WIDGET_GRAY, bg_colors=None, hover_color=YELLOW, extension_offset=(0, 0), anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, click_effect=False, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font, text)
        self.text = text
        self.current = text
        self.command = command
        self.text_color = text_color
        self.bg_color = bg_color
        self.bg_colors = bg_colors
        self.hover_color = hover_color
        self.click_effect = click_effect
        self.init_size(width, height, text)
        self.image.fill(bg_color)
        write(self.image, "center", self.text, self.font, text_color, *[s / 2 for s in self.image.get_size()])
        self.rect = self.image.get_rect()
        self.extension_offset = extension_offset
        # combos
        self.combo_width = max(font.size(combo)[0] + 6 for combo in combos)
        self.combo_height = sum(font.size(combo)[1] + 6 for combo in combos)
        self.combo_size = (self.combo_width, self.combo_height)
        self.combo_image = pygame.Surface((self.combo_width, self.combo_height))
        self.combo_image.fill(bg_color)
        self.combo_rects = {}
        x = y = 0
        for i, combo in enumerate(combos):
            w = self.combo_width
            h = font.size(combo)[1]
            combo_rect = pygame.Rect(self.rect.x + x, self.rect.y + y, w, h + 6)
            self.combo_image.fill(bg_color if bg_colors is None else bg_colors[i], combo_rect)
            write(self.combo_image, "topleft", combo, font, text_color, x + 3, y + 3)
            self.combo_rects[combo] = combo_rect
            y += combo_rect.height
        self.combo_image = Texture.from_surface(surf, self.combo_image)
        self.combo_rect = self.combo_image.get_rect(topleft=self.rect.topleft)
        self.real_combo_rects = None
        self.extended = False
        # last init
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def process_event(self, event):
        if is_left_click(event):
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.extended = not self.extended
            if self.extended and self.real_combo_rects is not None:
                for name, rect in self.real_combo_rects.items():
                    if rect.collidepoint(pygame.mouse.get_pos()):
                        self.overwrite(name)
                        self.extended = False
                        self.current = name
                        self.command(name)

    def update(self):
        if self.extended:
            self.reload_combo_rects()
            self.surf.blit(self.combo_image, self.real_combo_rect)
            for name, rect in self.real_combo_rects.items():
                if rect.collidepoint(pygame.mouse.get_pos()):
                    draw_rect(self.surf, self.hover_color, rect)

    def reload_combo_rects(self):
        xo, yo = self.extension_offset[0] * self.combo_width, self.extension_offset[1] * self.combo_height
        self.real_combo_rects = {name: pygame.Rect(self.rect.x + rect.x + xo, self.rect.y + rect.y + yo, *rect.size) for name, rect in self.combo_rects.items()}
        self.real_combo_rect = pygame.Rect(self.rect.x + self.combo_rect.x + xo, self.rect.y + self.combo_rect.y + yo, *self.combo_rect.size)

class ToggleButton(_Widget, ButtonBehavior, _Overwriteable):
    def __init__(self, surf, cycles, pos=_DEF_WIDGET_POS, command=None, width=None, height=None, bg_color=WIDGET_GRAY, text_color=BLACK, anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font)
        self.command = command
        self.bg_color = bg_color
        self.cycles = cycles
        self.cycle = 0
        self.text = self.cycles[self.cycle]
        text = self.text
        self.init_size(width, height, text)
        self.image.fill(self.bg_color)
        write(self.image, "center", text, self.font, text_color, *[s / 2 for s in self.image.get_size()])
        self.rect = self.image.get_rect()
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    @property
    def option(self):
        return self.cycles[self.cycle]

    def process_event(self, event):
        if is_left_click(event):
            mouse = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse):
                self.cycle += 1
                if self.cycle > len(self.cycles) - 1:
                    self.cycle = 0
                self.overwrite(self.cycles[self.cycle])
                if self.command is not None:
                    self.command(self.cycles[self.cycle])


class Label(_Widget, _Overwriteable):
    def __init__(self, surf, text, pos=_DEF_WIDGET_POS, width=None, height=None, bg_color=WIDGET_GRAY, text_color=BLACK, text_orien="center", anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font, text)
        self.bg_color = bg_color
        self.init_size(width, height, text)
        self.image.fill(bg_color)
        self.write_text(self.text, self.font, text_color, text_orien)
        self.rect = self.image.get_rect()
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)


class Entry(_Widget):
    def __init__(self, surf, title, command, width=None, height=None, max_chars=None, input_required=False, focus=True, keyboard=True, keyboard_map=None, key_font=None, joystick=None, func_args=None, text_color=BLACK, pos=_DEF_WIDGET_POS, start_command=None, anchor="center", default_text=None, exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, error_command=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font)
        self.text = title
        self.max_chars = max_chars if max_chars is not None else float("inf")
        self.input_required = input_required
        self.keyboard = keyboard
        self.keyboard_map = keyboard_map
        self.key_font = key_font if key_font is not None else _cfonts[20]
        self.joystick = joystick
        self.has_keyboard = False
        self.func_args = func_args
        self.text_color = text_color
        self.text_width = self.font.size(title)[0] + 10
        self.image = pygame.Surface((self.text_width, 60))
        self.image.fill(WIDGET_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        # title
        write(self.image, "center", title, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        self.rect = self.image.get_rect()
        self.command = command
        self.start_command = start_command
        self.error_command = error_command
        self.default_text = default_text
        self.height = 30
        self.output = ""
        if self.start_command:
            self.start_command()
        self.on = True
        self.last_on = ticks()
        self.focused = focus
        self.replacement = ""
        # super
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def init_keyboard(self):
        self.focused_key = "g"
        x = self.rect.x
        y = self.rect.y + self.rect.height
        xo = -(525 / 2 - self.rect.width / 2)
        yo = 0
        s = 35
        self.s = s
        self.keyboard_rects = {
            "!esc": (x, y, s, s), "1": (x + s * 1, y, s, s), "2": (x + s * 2, y, s, s), "3": (x + s * 3, y, s, s), "4": (x + s * 4, y, s, s), "5": (x + s * 5, y, s, s), "6": (x + s * 6, y, s, s), "7": (x + s * 7, y, s, s), "8": (x + s * 8, y, s, s), "9": (x + s * 9, y, s, s), "0": (x + s * 10, y, s, s), "-": (x + s * 11, y, s, s), "=": (x + s * 12, y, s, s), "!backspace": (x + s * 13, y, s * 2, s),
            "!tab": (x, y + s, s * 1.5, s), "q": (x + s * 1.5 + s * 0, y + s, s, s), "w": (x + s * 1.5 + s * 1, y + s, s, s), "e": (x + s * 1.5 + s * 2, y + s, s, s), "r": (x + s * 1.5 + s * 3, y + s, s, s), "t": (x + s * 1.5 + s * 4, y + s, s, s), "y": (x + s * 1.5 + s * 5, y + s, s, s), "u": (x + s * 1.5 + s * 6, y + s, s, s), "i": (x + s * 1.5 + s * 7, y + s, s, s), "o": (x + s * 1.5 + s * 8, y + s, s, s), "p": (x + s * 1.5 + s * 9, y + s, s, s), "{": (x + s * 1.5 + s * 10, y + s, s, s), "}": (x + s * 1.5 + s * 11, y + s, s, s), "!enter1": (x + s * 1.5 + s * 12, y + s, s * 1.5 + 1, s),
            "!caps lock": (x, y + s * 2, s * 1.75, s), "a": (x + s * 1.75 + s * 0, y + s * 2, s, s), "s": (x + s * 1.75 + s * 1, y + s * 2, s, s), "d": (x + s * 1.75 + s * 2, y + s * 2, s, s), "f": (x + s * 1.75 + s * 3, y + s * 2, s, s), "g": (x + s * 1.75 + s * 4, y + s * 2, s, s), "h": (x + s * 1.75 + s * 5, y + s * 2, s, s), "j": (x + s * 1.75 + s * 6, y + s * 2, s, s), "k": (x + s * 1.75 + s * 7, y + s * 2, s, s), "l": (x + s * 1.75 + s * 8, y + s * 2, s, s), ":": (x + s * 1.75 + s * 9, y + s * 2, s, s), "\"": (x + s * 1.75 + s * 10, y + s * 2, s, s), "|": (x + s * 1.75 + s * 11, y + s * 2, s, s), "!enter2": (x + s * 1.75 + s * 12, y + s * 2, s * 1.25 + 1, s),
            "!lshift": (x, y + s * 3, s * 2.25, s), "z": (x + s * 2.25 + s * 0, y + s * 3, s, s), "x": (x + s * 2.25 + s * 1, y + s * 3, s, s), "c": (x + s * 2.25 + s * 2, y + s * 3, s, s), "v": (x + s * 2.25 + s * 3, y + s * 3, s, s), "b": (x + s * 2.25 + s * 4, y + s * 3, s, s), "n": (x + s * 2.25 + s * 5, y + s * 3, s, s), "m": (x + s * 2.25 + s * 6, y + s * 3, s, s), "<": (x + s * 2.25 + s * 7, y + s * 3, s, s), ">": (x + s * 2.25 + s * 8, y + s * 3, s, s), "?": (x + s * 2.25 + s * 9, y + s * 3, s, s), "!rshift": (x + s * 2.25 + s * 10, y + s * 3, s * 2.75 + 1, s),
            "!lnothing": (x, y + s * 4, s * 2.25 + s * 1, s), " ": (x + s * 2.25 + s * 1, y + s * 4, s * 8, s), "!rnothing": (x + s * 2.25 + s * 1 + s * 8, y + s * 4, s * 3.75 + 1, s)
        }
        for key, rect in self.keyboard_rects.items():
            self.keyboard_rects[key] = pygame.Rect(rect[0] + xo, rect[1], rect[2] + yo, rect[3])
        self.min_x = min(rect.x for key, rect in self.keyboard_rects.items())
        self.min_y = min(rect.y for key, rect in self.keyboard_rects.items())
        self.wid_x = sum(rect.width for key, rect in self.keyboard_rects.items() if rect.y == y + yo)
        self.hei_y = sum(rect.height for key, rect in self.keyboard_rects.items() if rect.x == x + xo)
        self.has_keyboard = True

    def process_event(self, event):
        self.mod = pygame.key.get_mods()
        if event.type == pygame.KEYDOWN:
            self.key = pygame.key.name(event.key)
            if self.focused:
                if self.key == "return":
                    if self.output != "" if self.input_required else True:
                        self._exec_command(self.command, "out destroy", self.output, *(self.func_args if self.func_args else {}))
                        Thread(target=self.zoom, args=["out destroy"]).start()
                elif self.key == "backspace":
                    if self.mod == K_OSC:
                        self.output = self.output[:-1]
                        while self.output[-1:] != " " and self.output != "":
                            self.output = self.output[:-1]
                        self.output = self.output[:-1]
                    else:
                        self.output = self.output[:-1]
                elif self.key == "space":
                    self.inc_output(" ")
                else:
                    if len(self.key) == 1:
                        if len(self.output) < self.max_chars:
                            if self.mod == K_OSC and self.key == "v":
                                self.inc_output(get_clipboard())
                            else:
                                if self.mod == K_SHIFT:
                                    self.inc_output(keyboard_map.get(self.key, self.key.capitalize()))
                                else:
                                    self.inc_output(self.key)

        if event.type in (pygame.KEYDOWN, pygame.JOYBUTTONDOWN):
            try:
                frect = self.keyboard_rects[self.focused_key]
            except KeyError:
                pass
            except AttributeError:
                self.init_keyboard()
            else:
                if event.type == pygame.KEYDOWN:
                    conds = {"up": event.key == pygame.K_UP, "down": event.key == pygame.K_DOWN, "left": event.key == pygame.K_LEFT, "right": event.key == pygame.K_RIGHT}
                elif event.type == pygame.JOYBUTTONDOWN:
                    conds = {"up": event.button == self.keyboard_map["up"], "down": event.button == self.keyboard_map["down"], "left": event.button == self.keyboard_map["left"], "right": event.button == self.keyboard_map["right"]}
                if conds["up"]:
                    self.focused_key = sorted(self.keyboard_rects.items(), key=lambda x: (frect.y - x[1].y != self.s, abs(x[1].x - frect.x)))[0][0]
                elif conds["down"]:
                    self.focused_key = sorted(self.keyboard_rects.items(), key=lambda x: (x[1].y - frect.y != self.s, abs(x[1].x - frect.x)))[0][0]
                elif conds["left"]:
                    self.focused_key = sorted(self.keyboard_rects.items(), key=lambda x: (x[1].x >= frect.x, abs(x[1].y - frect.y), abs(x[1].x - frect.x)))[0][0]
                elif conds["right"]:
                    self.focused_key = sorted(self.keyboard_rects.items(), key=lambda x: (x[1].x <= frect.x, abs(x[1].y - frect.y), abs(x[1].x - frect.x)))[0][0]

        if is_left_click(event):
            mouse = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse):
                if not self.focused:
                    self.focused = True
                    for friend in self.friends:
                        friend.focused = False

    def inc_output(self, what):
        self.output += what
        self.on = True
        self.last_on = ticks()
        self.focused_key = what.lower()

    def update(self):
        mod = self.mod = pygame.key.get_mods()
        if self.keyboard and not self.has_keyboard and not self.zooming:
            self.init_keyboard()
        if self.has_keyboard and 1 == 0:
            for key, rect in self.keyboard_rects.items():
                if key == self.focused_key:
                    (self.surf, (180, 180, 180), rect)
                elif rect.collidepoint(pygame.mouse.get_pos()):
                    (self.surf, (240, 240, 240), rect)
                else:
                    (self.surf, (220, 220, 220), rect)
                (self.surf, (40, 40, 40), rect, 1)
                if not key.startswith("!"):
                    wr = keyboard_map.get(key, key.upper() if mod == K_SHIFT else key) if mod == K_SHIFT else key
                    write(self.surf, "center", wr, self.key_font, BLACK, *rect.center, tex=True)
            (self.surf, BLACK, (self.min_x, self.min_y, self.wid_x, self.hei_y), 1)
        w, h = self.image.width, self.image.height
        # self.image.fill(WHITE, (0, h / 2, w, h / 2))
        if self.default_text and self.output == "":
            write(self.surf, "midleft", self.default_text[0], self.default_text[1], WIDGET_GRAY, self.rect.x + 5, self.rect.y + self.image.height / 4 * 3, tex=True)
        elif self.on and self.focused:
            write(self.surf, "midleft", self.output + "|", self.font, self.text_color, self.rect.x + 5, self.rect.y + self.image.height / 4 * 3, tex=True)
        else:
            if self.output:
                write(self.surf, "midleft", self.output, self.font, self.text_color, self.rect.x + 5, self.rect.y + self.image.height / 4 * 3, tex=True)
        if ticks() - self.last_on >= 500:
            self.on = not self.on
            self.last_on = ticks()


class MessageboxOkCancel(_Widget):
    def __init__(self, surf, text, ok_command, no_ok_command=lambda_none, ok="OK", no_ok="CANCEL", width=None, height=None, pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, ok_joystick=None, *args, **kwargs):
        # base
        _Widget.__pre_init__(self, font, text)
        self.text_width = self.font.size(self.text)[0] + 10
        self.image = pygame.Surface((self.text_width, 60))
        self.image.fill(WIDGET_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.commands = {"ok": ok_command, "no_ok": no_ok_command}
        self.ok_joystick = ok_joystick
        # title
        write(self.image, "center", self.text, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        # ok
        write(self.image, "center", ok, self.font, BLACK, self.image.get_width() / 4, self.image.get_height() / 4 * 3)
        # cancel
        write(self.image, "center", no_ok, self.font, BLACK, self.image.get_width() / 4 * 3, self.image.get_height() / 4 * 3)
        # rects
        (self.image, BLACK, (self.image.get_width() // 2 - 2, 30, 4, 30))
        self.rects = {}
        self.rects["ok"] = pygame.Rect(self.rect.left, self.rect.top + 30, self.text_width // 2, 60)
        self.rects["no_ok"] = pygame.Rect(self.rect.left + self.text_width // 2, self.rect.top + 30, self.text_width, 60)
        # super
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == K_ENTER:
                self._exec_command(self.commands["ok"], "out destroy")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if is_left_click(event):
               for k, rect in self.rects.items():
                    if rect.collidepoint(mouse):
                        self._exec_command(self.commands[k], "out destroy")


class MessageboxOk(_Widget):
    def __init__(self, surf, text, width, height, ok_command=lambda_none, ok="OK", pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, ok_joystick=None, *args, **kwargs):
        # base
        _Widget.__pre_init__(self, font, text)
        self.text_width = self.font.size(self.text)[0] + 10
        self.image = pygame.Surface((self.text_width, 60))
        self.image.fill(WIDGET_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.commands = {"ok": ok_command}
        self.ok_joystick = ok_joystick
        # title
        write(self.image, "center", self.text, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        # ok
        write(self.image, "center", "OK", self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4 * 3)
        # rects
        self.rects = {}
        self.rects["ok"] = pygame.Rect(self.rect.left, self.rect.top + 30, self.text_width, 60)
        # super
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == K_ENTER:
                self._exec_command(self.commands["ok"], "out destroy")

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if is_left_click(event):
               for k, rect in self.rects.items():
                    if rect.collidepoint(mouse):
                        self._exec_command(self.commands[k], "out destroy")

        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == self.ok_joystick:
                self._exec_command(self.commands["ok"], "out destroy")


class MessageboxError(_Widget):
    def __init__(self, surf, text, pos=_DEF_WIDGET_POS, anchor="center", width=None, height=None, exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, ok_joystick=None, *args, **kwargs):
        _Widget.__pre_init__(self, font, text)
        self.text_width = self.font.size(self.text)[0] + 10
        self.image = pygame.Surface((self.text_width, 60))
        self.image.fill(WIDGET_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.ok_joystick = ok_joystick
        # title
        write(self.image, "center", self.text, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        # ok
        write(self.image, "center", "OK", self.font, BLACK, self.image.get_width() / 4 * 3, self.image.get_height() / 4 * 3)
        self.rects = {}
        self.rects["ok"] = pygame.Rect(self.rect.left + self.text_width // 2, self.rect.top + 30, self.text_width, 60)
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == K_ENTER:
                Thread(target=self.zoom, args=["out destroy"]).start()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if is_left_click(event):
                if self.rects["ok"].collidepoint(mouse):
                    self._exec_command(self.exit_command, "out destroy")

        elif event.type == pygame.JOYBUTTONDOWN:
            if event.button == self.ok_joystick:
                self._exec_command(self.commands["ok"], "out destroy")


class Checkbox(_Widget, _Overwriteable):
    def __init__(self, surf, text, while_checked_command=None, check_command=None, uncheck_command=None, while_not_checked_command=None, width=None, height=None, checked=False, pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font)
        self.text = text
        self.bg_color = WIDGET_GRAY
        w = width - 30 if width is not None else self.font.size(text)[0]
        h = height - 10 if height is not None else self.font.size(text)[1]
        self.image = pygame.Surface((30 + 5 + w + 5, 5 + h + 5))
        self.image.fill(self.bg_color)
        write(self.image, "topleft", text, self.font, BLACK, 30, 5)
        self.box = pygame.Surface((h - 5, h - 5))
        self.box.fill(WHITE)
        self.box = Texture.from_surface(surf, self.box)
        self.rect = self.image.get_rect(center=pos)
        self.rects = {}
        self.while_checked_command = while_checked_command
        self.while_not_checked_command = while_not_checked_command
        self.check_command = check_command
        self.uncheck_command = uncheck_command
        self.check = get_icon("check", (h - 5, h - 5))
        self.check = Texture.from_surface(surf, self.check)
        self.checked = False
        if checked:
            self.check_event()
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    def __bool__(self):
        return self.checked

    def process_event(self, event):
        if is_left_click(event):
            mouse = pygame.mouse.get_pos()
            if pygame.Rect(self.rect.left + 5, self.rect.top + 5, 20, 20).collidepoint(mouse):
                self.check_event()

    def update(self):
        bw, bh = self.box.width, self.box.height
        xo, yo = 5, self.image.height / 2 - self.box.height / 2
        blitx, blity = self.rect.x + xo, self.rect.y + yo
        rect = pygame.Rect(blitx, blity, bw, bh)
        self.surf.blit(self.box, rect)
        if self.checked:
            self.surf.blit(self.check, rect)
            self._exec_command(self.while_checked_command)
        elif self.while_not_checked_command:
            self._exec_command(self.while_not_checked_command)

    def check_event(self):
        self.checked = not self.checked
        if self.checked and self.check_command:
            self._exec_command(self.check_command)
        elif not self.checked and self.uncheck_command:
            self._exec_command(self.uncheck_command)


class Slider(_Widget):
    def __init__(self, surf, text, values, start=None, on_move_command=None, decimals=0, pos=_DEF_WIDGET_POS, anchor="center", color=WIDGET_GRAY, width=None, height=None, exit_command=None, visible_when=None, font=None, tooltip_font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, tooltip=None, appends=None, as_child=False, *args, **kwargs):
        _Widget.__pre_init__(self, font, text)
        self.on_move_command = on_move_command
        fs = self.font.size(text)
        daw_ = 40
        self.text = text
        self.init_size(width, height, text)
        self.color = color
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.values = values
        try:
            self.value = values[start] if start is not None else values[0]
        except IndexError:
            self.value = start if start is not None else values[0]
        self.range = self.image.get_width() - 17
        self.ratio = (len(self.values) - 1) / self.range
        self.mult = self.range / (len(self.values) - 1)
        self.pressed = False
        _Widget.__init__(self, self.image, surf, visible_when, friends, pos, anchor, width, height, exit_command, disabled, disable_type, template, type(self), add, special_flags, tooltip, appends, as_child)

    @property
    def tooltip_rect(self):
        return pygame.Rect(*self.rect.topleft, self.rect.width, self.rect.height / 2)

    def in_area(self):
        return pygame.mouse.get_pressed()[0] and 5 <= self.mouse[0] <= self.rect.width - 5 and 8 <= self.mouse[1] <= self.rect.height - 8

    def process_event(self, event):
        prev_value = self.value
        if is_left_click(event):
            self.pressed = self.in_area()
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.pressed = False
        if event.type == pygame.MOUSEMOTION:
            if pygame.mouse.get_pressed()[0]:
                self.update()
                if prev_value != self.value:
                    self._exec_command(self.on_move_command, None, self.value)

    def update(self):
        self.mouse = pygame.mouse.get_pos()
        self.mouse = (self.mouse[0] - self.rect.x, self.mouse[1] - self.rect.y)
        self.mouses = pygame.mouse.get_pressed()
        if len(threading.enumerate()) == 1:
            if not hasattr(self, "slider_img"):
                self.slider_img = pygame.Surface((7, self.image.height / 2 - 4))
                self.slider_img.fill(GRAY)
                self.slider_img = Texture.from_surface(self.surf, self.slider_img)
            if not hasattr(self, "slider_rect"):
                self.slider_rect = self.slider_img.get_rect(bottomleft=(5, self.image.height - 8))
                with suppress(ValueError):
                    self.slider_rect.x += self.values.index(self.value if isinstance(self.value, str) else round(self.value)) * self.mult
        if hasattr(self, "slider_img") and hasattr(self, "slider_rect"):
            if hasattr(self, "pressed") and self.pressed:
                self.slider_rect.centerx = self.mouse[0]
                if self.slider_rect.left < 5:
                    self.slider_rect.left = 5
                elif self.slider_rect.right > self.rect.width - 5:
                    self.slider_rect.right = self.rect.width - 5
                self.value = self.values[round((self.slider_rect.x - 5) * self.ratio)]
            # self.image.fill(self.color)
            write(self.surf, "topleft", self.text, self.font, BLACK, self.rect.x + 5, self.rect.y + 5, tex=True)
            write(self.surf, "topright", self.value, self.font, BLACK, self.rect.x + self.image.width - 5, self.rect.y, tex=True)
            self.surf.blit(self.slider_img, self.slider_rect)


def update_and_poll_widgets():
    ret = []
    tooltip_data = []
    for widget in iter_widgets():
        if widget.visible_when is None:
            if hasattr(widget, "draw") and not widget.disabled:
                ret.append((widget, widget.image, widget.rect))
            if hasattr(widget, "update"):
                widget.update()
        else:
            if widget.visible_when():
                ret.append((widget, widget.image, widget.rect))
        if widget.tooltip is not None:
            if not widget.disabled:
                if not hasattr(widget, "last_hover"):
                    widget.last_hover = ticks()
                else:
                    if getattr(widget, "tooltip_rect", widget.rect).collidepoint(pygame.mouse.get_pos()):
                        if ticks() - widget.last_hover >= 750:
                            mp = pygame.mouse.get_pos()
                            td = [widget.tooltip_img, (mp[0], mp[1] - widget.tooltip_img.get_height()), widget.surf]
                            tooltip_data.append(td)
                    else:
                        widget.last_hover = ticks()
    # for img, pos, dest in tooltip_data:
    #     dest.blit(img, pos)
    ret += [[None] + x[0:2] for x in tooltip_data]
    return ret
    for cursor, cond in _eng.cursors:
        if cond() and pygame.mouse.get_cursor() != cursor:
            pygame.mouse.set_cursor(cursor)
            break


def format_text(text):
    return text.replace("\t", "    ")


def set_default_fonts(font):
    _eng.def_fonts = font


def set_default_tooltip_fonts(font):
    _eng.def_tooltip_fonts = font


def set_cursor_when(cursor, when):
    _eng.cursors.append((cursor, when))


def update_button_behavior(itr):
    update = False
    mouse = pygame.mouse.get_pos()
    for i, widget in enumerate(itr + iter_widgets()):
        if not isinstance(widget, _Widget) or isinstance(widget, ButtonBehavior):
            if widget.rect.collidepoint(mouse):
                widget.image.set_alpha(150)
            else:
                widget.image.set_alpha(255)


def befriend_iterable(itr):
    for owidget in itr:
        for iwidget in itr:
            if owidget is not iwidget:
                owidget.add_friend(iwidget)


def no_widgets(type_=None):
    return not any((widget for widget in iter_widgets() if isinstance(widget, type_ if type_ is not None else _Widget) and not widget.disabled))


def iter_overwriteables():
    return [widget for widget in iter_widgets() if hasattr(widget, "overwrite")]


def len_enabled_widgets():
    return len([widget for widget in iter_widgets() if not widget.disabled])


def iter_widgets(type_=_Widget):
    return sorted([widget for widget in _mod.widgets if isinstance(widget, type_)], key=lambda widget: isinstance(widget, Entry))


def iter_particles():
    return _mod.particles


def iter_buttons():
    return [widget for widget in iter_widgets() if isinstance(widget, (Button, ToggleButton))]


def destroy_widgets():
    for widget in iter_widgets():
        if widget.visible_when is None:
            if widget.disable_type == "disable":
                widget.disable()
            elif widget.disable_type == "destroy":
                widget.destroy()


def draw_and_update_widgets():
    tooltip_data = []
    for widget in iter_widgets():
        if widget.visible_when() if widget.visible_when is not None else True:
            if hasattr(widget, "draw") and not widget.disabled:
                widget.draw()
            if hasattr(widget, "update") and not widget.disabled:
                widget.update()
        if widget.tooltip is not None:
            if not widget.disabled:
                if not hasattr(widget, "last_hover"):
                    widget.last_hover = ticks()
                else:
                    if getattr(widget, "tooltip_rect", widget.rect).collidepoint(pygame.mouse.get_pos()):
                        if ticks() - widget.last_hover >= 750:
                            mp = pygame.mouse.get_pos()
                            td = [widget.tooltip_img, (mp[0], mp[1] - widget.tooltip_img.get_height()), widget.surf]
                            tooltip_data.append(td)
                    else:
                        widget.last_hover = ticks()
    # for img, pos, dest in tooltip_data:
    #     dest.blit(img, pos)
    for cursor, cond in _eng.cursors:
        if cond() and pygame.mouse.get_cursor() != cursor:
            pygame.mouse.set_cursor(cursor)
            break


def process_widget_events(event, mouse):
    for widget in iter_widgets():
        if hasattr(widget, "process_event") and callable(widget.process_event):
            if not widget.disabled:
                widget.process_event(event)


class _Engine:
    def __init__(self):
        self.def_cursor = None
        self.def_fonts = _cfonts
        self.def_tooltip_fonts = None
        self.cursors = []
        self.widget_startup_commands = []


class _Module:
    def __init__(self):
        self.widgets = []
        self.particles = []


_eng = _Engine()
_mod = _Module()
