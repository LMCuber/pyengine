from .basics import *
from .pgbasics import *
from .pilbasics import *


_DEF_WIDGET_POS = (100, 100)
WOOSH_TIME = 0.001
_cfonts = []
for i in range(101):
    _cfonts.append(pygame.font.SysFont("calibri", i))
keyboard_map = {"1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*", "9": "(", "0": ")", "-": "_", "=": "+"}


def set_default_fonts(font):
    _eng.def_fonts = font


def set_cursor_when(cursor, when):
    _eng.cursors.append((cursor, when))


def update_button_behavior(itr):
    update = False
    mouse = pygame.mouse.get_pos()
    for i, widget in enumerate(itr + iter_widgets()):
        if not isinstance(widget, _Widget) or isinstance(widget, BaseButton):
            if widget.rect.collidepoint(mouse):
                widget.image.set_alpha(150)
            else:
                widget.image.set_alpha(255)


def befriend_iterable(itr):
    for focused_widget in itr:
        for looping_widget in itr:
            if focused_widget is not looping_widget:
                focused_widget.add_friend(looping_widget)


def no_widgets(type_=None):
    return not any((widget for widget in iter_widgets() if isinstance(widget, type_ if type_ is not None else _Widget) and not widget.disabled))


def iter_widgets():
    return sorted(_mod.widgets, key=lambda widget: isinstance(widget, Entry))


def len_enabled_widgets():
    return len([widget for widget in iter_widgets() if not widget.disabled])


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
    for widget in iter_widgets():
        if widget.visible_when is None:
            if hasattr(widget, "draw") and not widget.disabled:
                widget.draw()
            if hasattr(widget, "update"):
                widget.update()
        else:
            if widget.visible_when():
                widget.draw()


def process_widget_events(event, mouse):
    for widget in iter_widgets():
        if hasattr(widget, "process_event") and callable(widget.process_event):
            if not widget.disabled:
                widget.process_event(event)


class _Engine:
    def __init__(self):
        self.def_cursor = None
        self.def_fonts = _cfonts
        self.cursors = []
        self.widget_startup_commands = []


class _Module:
    def __init__(self):
        self.widgets = []
        self.particles = []


_eng = _Engine()
_mod = _Module()


class _Widget:
    def __init__(self, image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, child, add, special_flags, *args, **kwargs):
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
                self.friends.append(friend)
                friend.add_friend(self)
        if add:
            _mod.widgets.append(self)
        self.child = child
        if "rounded" in self.special_flags:
            img = pil_to_pg(round_corners(pg_to_pil(self.image), 10))
            self.image = img
            self.replace_og(img)
        self.visible_when = visible_when
        setattr(self.rect, anchor, pos)
        Thread(target=self.zoom, args=["in"]).start()
        self.startup_command()

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
        Thread(target=self.zoom, args=["in"]).start()
        self.disabled = False
        self.startup_command()

    def disable(self):
        Thread(target=self.zoom, args=["out"]).start()

    def zoom(self, type_):
        if type_ == "in":
            for i in range(20):
                size = [round(fromperc((i + 1) * 5, s)) for s in self.og_size]
                self.image = pygame.transform.scale(self.og_img, size)
                self.rect = self.image.get_rect()
                setattr(self.rect, self.og_anchor, self.og_pos)
                time.sleep(WOOSH_TIME)
        elif "out" in type_:
            for i in reversed(range(20)):
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

    def draw(self):
        self.surf.blit(self.image, self.rect)

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
            self.image.set_alpha(150)
        else:
            self.image.set_alpha(255)


class _Overwriteable:
    def overwrite(self, text):
        w, h = [5 + size + 5 for size in self.font.size(str(text))]
        self.image = pygame.Surface((w, h)).convert_alpha()
        self.image.fill(self.bg_color)
        write(self.image, "center", text, self.font, BLACK, *[s / 2 for s in self.image.get_size()])
        self.replace_og(self.image)
        self.rect = self.image.get_rect(center=self.rect.center)


class BaseButton:
    pass


class Button(_Widget, BaseButton):
    def __init__(self, surf, text, command, width=None, height=None, pos=_DEF_WIDGET_POS, text_color=BLACK, bg_color=LIGHT_GRAY, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, click_effect=False, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.text = text
        self.font = font if font is not None else _eng.def_fonts[20]
        self.bg_color = bg_color
        self.click_effect = click_effect
        w = width if width is not None else self.font.size(text)[0] + 5
        h = height if height is not None else self.font.size(text)[1] + 5
        self.image = pygame.Surface((w, h)).convert_alpha()
        self.image.fill(bg_color)
        write(self.image, "center", text, self.font, text_color, *[s / 2 for s in self.image.get_size()])
        self.rect = self.image.get_rect()
        self.command = command
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

    def process_event(self, event):
        mouse = pygame.mouse.get_pos()
        if is_left_click(event):
            if self.rect.collidepoint(mouse):
                self.command()
                if self.click_effect:
                    Thread(target=self.zoom, args=["out"]).start()


class ToggleButton(_Widget, BaseButton, _Overwriteable):
    def __init__(self, surf, cycles, pos=_DEF_WIDGET_POS, command=None, bg_color=LIGHT_GRAY, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.font = font if font is not None else _eng.def_fonts[20]
        self.command = command
        self.bg_color = bg_color
        self.cycles = cycles
        self.cycle = 0
        w, h = [5 + size + 5 for size in self.font.size(str(self.cycles[self.cycle]))]
        self.image = pygame.Surface((w, h)).convert_alpha()
        self.image.fill(self.bg_color)
        write(self.image, "topleft", self.cycles[self.cycle], self.font, BLACK, 5, 5)
        self.rect = self.image.get_rect()
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

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
    def __init__(self, surf, text, pos=_DEF_WIDGET_POS, bg_color=LIGHT_GRAY, text_color=BLACK, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.text = text
        self.font = font if font is not None else _eng.def_fonts[20]
        self.bg_color = bg_color
        w, h = [s + 5 for s in self.font.size(text)]
        self.image = pygame.Surface((w, h)).convert_alpha()
        self.image.fill(bg_color)
        write(self.image, "center", text, self.font, text_color, *[s / 2 for s in self.image.get_size()])
        self.rect = self.image.get_rect()
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)


class Entry(_Widget):
    def __init__(self, surf, title, command, max_chars=None, input_required=False, focus=True, pos=_DEF_WIDGET_POS, start_command=None, anchor="center", default_text=None, exit_command=None, visible_when=None, font=None, friends=None, error_command=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.font = font if font is not None else _eng.def_fonts[20]
        self.max_chars = max_chars if max_chars is not None else float("inf")
        self.input_required = input_required
        self.text_width = self.font.size(title)[0] + 10
        self.image = pygame.Surface((self.text_width, 60)).convert_alpha()
        self.image.fill(LIGHT_GRAY)
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
        self.x = 5
        self.output = ""
        if self.start_command:
            self.start_command()
        self.on = True
        self.last_on = ticks()
        self.focused = focus
        self.replacement = ""
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            key = pygame.key.name(event.key)
            mod = pygame.key.get_mods()
            if self.focused:
                if key == "return":
                    if self.output != "" if self.input_required else True:
                        self._exec_command(self.command, "out destroy", self.output)
                        Thread(target=self.zoom, args=["out destroy"]).start()
                elif key == "backspace":
                    if mod == K_CTRL:
                        self.output = self.output[:-1]
                        while self.output[-1:] != " " and self.output != "":
                            self.output = self.output[:-1]
                        self.output = self.output[:-1]
                    else:
                        self.output = self.output[:-1]
                elif key == "space":
                    self.inc_output(" ")
                else:
                    if len(key) == 1:
                        if len(self.output) < self.max_chars:
                            if mod == K_CTRL and key == "v":
                                self.inc_output(get_clipboard())
                            else:
                                if mod == K_SHIFT:
                                    self.inc_output(keyboard_map.get(key, key.capitalize()))
                                else:
                                    self.inc_output(key)

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

    def update(self):
        w, h = self.image.get_width(), self.image.get_height()
        self.image.fill(WHITE, (0, h / 2, w, h / 2))
        if self.default_text and self.output == "":
            write(self.image, "midleft", self.default_text[0], self.default_text[1], LIGHT_GRAY, 5, self.image.get_height() / 4 * 3)
        elif self.on and self.focused:
            write(self.image, "midleft", self.output + "|", self.font, BLACK, 5, self.image.get_height() / 4 * 3)
        else:
            write(self.image, "midleft", self.output, self.font, BLACK, 5, self.image.get_height() / 4 * 3)
        if ticks() - self.last_on >= 500:
            self.on = not self.on
            self.last_on = ticks()


class MessageboxOkCancel(_Widget):
    def __init__(self, surf, text, ok_command, no_ok_command=lambda: None, ok="OK", no_ok="CANCEL", pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        # base
        self.font = font if font is not None else _eng.def_fonts[20]
        self.text_width = self.font.size(text)[0] + 10
        self.image = pygame.Surface((self.text_width, 60)).convert_alpha()
        self.image.fill(LIGHT_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        self.commands = {"ok": ok_command, "no_ok": no_ok_command}
        # title
        write(self.image, "center", text, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        # ok
        write(self.image, "center", "OK", self.font, BLACK, self.image.get_width() / 4, self.image.get_height() / 4 * 3)
        # cancel
        write(self.image, "center", "CANCEL", self.font, BLACK, self.image.get_width() / 4 * 3, self.image.get_height() / 4 * 3)
        # rects
        pygame.draw.rect(self.image, BLACK, (self.image.get_width() // 2 - 2, 30, 4, 30))
        self.rects = {}
        self.rects["ok"] = pygame.Rect(self.rect.left, self.rect.top + 30, self.text_width // 2, 60)
        self.rects["no_ok"] = pygame.Rect(self.rect.left + self.text_width // 2, self.rect.top + 30, self.text_width, 60)
        # super
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

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


class MessageboxError(_Widget):
    def __init__(self, surf, text, pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.font = font if font is not None else _eng.def_fonts[20]
        self.text_width = self.font.size(text)[0] + 10
        self.image = pygame.Surface((self.text_width, 60)).convert_alpha()
        self.image.fill(LIGHT_GRAY)
        under = pygame.Surface((self.text_width, 30))
        under.fill(WHITE)
        self.image.blit(under, (0, 30))
        self.rect = self.image.get_rect()
        self.rect.center = pos
        # title
        write(self.image, "center", text, self.font, BLACK, self.image.get_width() / 2, self.image.get_height() / 4)
        # ok
        write(self.image, "center", "OK", self.font, BLACK, self.image.get_width() / 4 * 3, self.image.get_height() / 4 * 3)
        self.rects = {}
        self.rects["ok"] = pygame.Rect(self.rect.left + self.text_width // 2, self.rect.top + 30, self.text_width, 60)
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

    def process_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == K_ENTER:
                Thread(target=self.zoom, args=["out destroy"]).start()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            if is_left_click(event):
                if self.rects["ok"].collidepoint(mouse):
                    self._exec_command(self.exit_command, "out destroy")


class Checkbox(_Widget):
    def __init__(self, surf, text, while_checked=None, check_command=None, uncheck_command=None, while_not_checked=None, width=None, height=None, checked=False, pos=_DEF_WIDGET_POS, anchor="center", exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.font = font if font is not None else _eng.def_fonts[20]
        w = width if width is not None else self.font.size(text)[0]
        h = height if height is not None else self.font.size(text)[1]
        self.image = pygame.Surface((30 + 5 + w + 5, 5 + h + 5)).convert_alpha()
        self.image.fill(LIGHT_GRAY)
        write(self.image, "topleft", text, self.font, BLACK, 30, 5)
        self.box = pygame.Surface((h - 5, h - 5))
        self.box.fill(WHITE)
        self.rect = self.image.get_rect(center=pos)
        self.checked = checked
        self.rects = {}
        self.while_checked = while_checked
        self.while_not_checked = while_not_checked
        self.check_command = check_command
        self.uncheck_command = uncheck_command
        self.check = get_icon("check", (h - 5, h - 5))
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

    def process_event(self, event):
        if is_left_click(event):
            mouse = pygame.mouse.get_pos()
            if pygame.Rect(self.rect.left + 5, self.rect.top + 5, 20, 20).collidepoint(mouse):
                self.checked = not self.checked
                if self.checked and self.check_command:
                    self._exec_command(self.check_command)
                elif not self.checked and self.uncheck_command:
                    self._exec_command(self.uncheck_command)

    def update(self):
        blitx, blity = 5, self.image.get_height() / 2 - self.box.get_height() / 2
        self.image.blit(self.box, (blitx, blity))
        if self.checked:
            bw, bh = self.box.get_size()
            self.image.blit(self.check, (blitx, blity))
            self._exec_command(self.while_checked)
        elif self.while_not_checked:
            self._exec_command(self.while_not_checked)


class Slider(_Widget):
    def __init__(self, surf, text, values, start=None, on_move_command=None, decimals=0, pos=(810 / 2, 600 / 2), anchor="center", color=LIGHT_GRAY, height=None, exit_command=None, visible_when=None, font=None, friends=None, disabled=False, disable_type=False, template=None, add=True, special_flags=None, *args, **kwargs):
        self.font = font if font is not None else _eng.def_fonts[20]
        self.on_move_command = on_move_command
        self.text = text
        fs = self.font.size(text)
        daw_ = 40
        self.image = pygame.Surface((5 + fs[0] + daw_, 5 + fs[1] + 5) if height is None else (5 + fs[0] + daw_, height)).convert_alpha()
        self.color = color
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.values = values
        self.value = start if start is not None else values[0]
        self.range = self.image.get_width() - 17
        self.ratio = (len(self.values) - 1) / self.range
        self.mult = self.range / (len(self.values) - 1)
        self.pressed = False
        super().__init__(self.image, surf, visible_when, friends, pos, anchor, exit_command, disabled, disable_type, template, type(self), add, special_flags)

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
                self.slider_img = pygame.Surface((7, self.image.get_height() / 2 - 4))
                self.slider_img.fill(GRAY)
            if not hasattr(self, "slider_rect"):
                self.slider_rect = self.slider_img.get_rect(bottomleft=(5, self.image.get_height() - 8))
                self.slider_rect.x += self.values.index(self.value if isinstance(self.value, str) else round(self.value)) * self.mult
        if hasattr(self, "slider_img") and hasattr(self, "slider_rect"):
            if hasattr(self, "pressed") and self.pressed:
                self.slider_rect.centerx = self.mouse[0]
                if self.slider_rect.left < 5:
                    self.slider_rect.left = 5
                elif self.slider_rect.right > self.rect.width - 5:
                    self.slider_rect.right = self.rect.width - 5
                self.value = self.values[round((self.slider_rect.x - 5) * self.ratio)]
            self.image.fill(self.color)
            write(self.image, "topleft", self.text, self.font, BLACK, 5, 5)
            write(self.image, "topright", self.value, self.font, BLACK, self.image.get_width() - 5, 5)
            self.image.blit(self.slider_img, self.slider_rect)
