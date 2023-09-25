from .imports import *
import translatepy
from translatepy.translators.google import GoogleTranslate as _TpyGoogleTranslate
from googletrans import Translator as _GoogleGoogleTranslate
from pandas.io.clipboard import clipboard_get as pd_clipboard_get
import operator as op
import numpy as np
import inspect
import sys
import os
import io
import random
import platform
import time
import requests
import string
import json
import inspect
import subprocess
import wave
import socket
from line_profiler import LineProfiler


# constants
char_mods = {"a": "áàâãä",
             "e": "éèêë",
             "i": "íìîï",
             "o": "óòôõö",
             "u": "úùûü"}
INF = "\u221e"  # infinity
DEG = "\u00B0"  # celcius
BULLET = "⁍"
int_to_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
pritn = print   # be cau sè
prrint = print  # be cau sè
pirnt = print   # be cau sè
PRINT = print   # be cau sè
pint = print    # be cau sè
epoch = time.time
lambda_none = lambda *a, **kwa: None
lambda_ret = lambda x: x
funny_words = {"lmao", "lmoa", "lol", "lol get rekt"}
gf_combo_names = ["super", "power", "ninja", "turbo", "neo", "ultra", "hyper", "mega", "multi", "alpha", "meta", "extra", "uber", "prefix"]
steel_colors = []
k_b = 1.380649 * 10 ** -23
n_a = 6.02214076 * 10 ** 23
gas_constant = k_b * n_a


# functions
def genlog(t, a, k, c, q, b, v):
    return a + (k - a) / (c + q * e ** (-b * t)) ** (1 / v)


def sustainability(km_per_l, rho, molar_mass, ratio):
    kg = 1 / km_per_l * rho
    moles = (kg * 1000) / molar_mass
    moles *= ratio
    return moles


def dot_product_3d(p1, p2):
    ab = p1[0] * p2[0] + p1[1] * p2[1] + p1[2] * p2[2]
    mag1 = sqrt(p1[0] ** 2 + p1[1] ** 2 + p1[2] ** 2)
    mag2 = sqrt(p2[0] ** 2 + p2[1] ** 2 + p2[2] ** 2)
    theta = acos(ab / (mag1 * mag2)) * (180 / pi)
    return theta


def diff(src, dest):
    return (dest[0] - src[0], dest[1] - src[1])


def semicircle(x):
    return sqrt(1 - x ** 2)


def rand_sine_wave(l):
    def inner(x):
        f = "".join([f"{i} * sin(x / {i})" + (" + " if i < l - 1 else "") for i in range(1, l)])
        ev = f.replace("x", str(x))
        return eval(ev)

    return inner


def delay(func, secs, *args, **kwargs):
    def inner():
        sleep(secs)
        func(*args, **kwargs)

    Thread(target=inner).start()


def rot_matrix_2d(x, y, p):
    xp = x * cos(p) - y * sin(p)
    yp = y * cos(p) + x * sin(p)
    return xp, yp


def trig_wave(a, w, h, t, x=0, y=0):
    ret = [0, 0]
    ret[0] = w * cos(a) * cos(t) - h * sin(a) * sin(t) + x
    ret[1] = w * cos(a) * sin(t) + h * sin(a) * cos(t) + y
    return ret


def lemniscate(x, a=1, p=1):  # x: function x; a: magnitude parameter; p: power
    return a * x * sqrt(1 - x ** 2) ** p


def flatten2dl(l):
    return sum(l, [])


def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    elif x < 0:
        return -1


def rand_alien_name():
    vs = "aeiou"
    abs_vs = vs + "".join(char_mods.values())
    cs = "bcdfghjklmnpqrstvwxyz"
    diff_js = {("t", "i"): "chi", ("s", "i"): "shi", ("t", "u"): "tsu", ("h", "u"): "fu"}
    js = [x for x in ("".join((f, s)) if (f, s) not in diff_js else diff_js[f, s] for (s, f) in product(vs, cs)) if x[0] not in ("c", "q", "x") and x not in ("wu", "yi", "ye")]
    ret = ""
    ch = 7 / 8
    for _ in range(rand(5, 9)):
        if ret:
            if ret[-1] in abs_vs:
                app = choice(cs if chance(ch) else vs)
            elif ret[-1] in cs:
                app = choice(vs if chance(ch) else cs)
        else:
            app = choice(vs if chance(1 / 2) else cs)
        if app in vs and chance(1 / 10):
            app = choice(char_mods[app])
        ret += app
    # Don / Dona
    if chance(1 / 20):
        ret += "us" if chance(1 / 2) else "a"
        ret = ("Don " if ret.endswith("us") else "Dona ") + ret
    # .title()
    ret = ret.title()
    # Sama
    if chance(1 / 20):
        ret += choice(js)
        if chance(1 / 15):
            ret += "n"
        ret += "-sama"
    return ret


def sec(x):
    return 1 / cos(x)


def csc(x):
    return 1 / sin(x)


def cot(x):
    return 1 / tan(x)


def variance(l):
    return sum((x - sum(l) / len(l)) ** 2 for x in l) / len(l)


def stdev(l):
    return sqrt(variance(l))


def audio_length(p):
    with wave.open(p, "r") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / rate
        return duration


def quadratic(a, b, c):
    def func(op_):
        return (op_(-b, sqrt(b ** 2 - 4 * a * c))) / 2 * a
    return (func(op.add), func(op.sub))


def is_normal(s):
    for char in s:
        if char not in string.ascii_lowercase + string.ascii_uppercase + string.punctuation + string.digits + "".join(char_mods.values()) + " ":
            return False
    return True


def bubble_sort(arr, func=lambda x, y: x > y, iterations=float("inf")):
    ret = deepcopy(arr)
    if isinstance(ret, list):
        dtype = "list"
    elif isinstance(ret, str):
        ret = list(ret)
        dtype = "str"
    n = len(arr)
    itr = 0
    for i in range(n - 1):
        for j in range(n - i - 1):
            if func(ret[j], ret[j + 1]):
                ret[j], ret[j + 1] = ret[j + 1], ret[j]
        itr += 1
        if iter == iterations:
            break
    if dtype == "list":
        return ret
    elif dtype == "str":
        return "".join(ret)


def shake_str(s):
    return bubble_sort(s, lambda x, y: chance(1 / 4), 1)


def chain_dict(l):
    ret = {}
    for e in l:
        for k, v in e.items():
            ret[k] = v
    return ret


def raise_type(argname, argindex, right, wrong):
    raise TypeError(f"Argument {argindex} ({argname}) must be of type '{right}', not '{wrong}'")


def dict_to_str(d):
    return str(d).removeprefix("{").removesuffix("}").replace("'", "").replace('"', '')


def osascript(script):
    if Platform.os == "darwin":
        os.system(f"osascript -e '{script}'")


def wchoice(c, w):
    return random.choices(c, weights=w)[0]


def choose_file(title, file_types="all"):
    s = os.path.sep
    ot = f' of type "{file_types}"' if file_types != "all" else ""
    if Platform.os != "darwin":
        pth = tkinter.filedialog.askopenfilename(title=title)
        return pth
    else:
        proc = subprocess.Popen(["osascript", "-e", f'choose file with prompt "{title}"{ot}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        comm = proc.communicate()
        pth = s + s.join(comm[0].decode().removeprefix("alias ").removesuffix("\n").replace(":", s).split(s)[1:])
        return pth


def choose_folder(title):
    s = os.path.sep
    if Platform.os != "darwin":
        pth = tkinter.filedoalog.askopenfilename(title=title, **kw)
    else:
        proc = subprocess.Popen(["osascript", "-e", f'choose folder with prompt "{title}"'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        comm = proc.communicate()
        pth = s + s.join(comm[0].decode().removeprefix("alias ").removesuffix("\n").replace(":", s).split(s)[1:])
        return pth


def pascal(rows):
    if rows == 1:
        return [[1]]
    triangle = [[1], [1, 1]]
    row = [1, 1]
    for i in range(2, rows):
        row = [1] + [sum(column) for column in zip(row[1:], row)] + [1]
        triangle.append(row)
    return triangle


def pyramid(height, item=0):
    ret = []
    for _ in range(height):
        try:
            ret.append([item] * (len(ret[-1]) + 1))
        except IndexError:
            ret.append([item])
    return ret


def do_nothing():
    pass


def open_text(filepath):
    if Platform.os == "windows":
        os.system(f"notepad {filepath}")
    elif Platform.os == "darwin":
        os.system(f"open {filepath}")


def ndget(nd, keys):
    for e in keys:
        nd = nd[e]
    return nd


def empty_function():
    return lambda *_, **__: None


def sassert(cond):
    if not cond:
        raise ArbitraryException


def create_exception(name, *parents):
    return type(name, tuple([*parents]), {})


def pin():
    return [choice(range(9)) for _ in range(4)]


def token(length=20, valid_chars="default"):
    chars = string.ascii_uppercase + string.ascii_lowercase + "".join([str(x) for x in range(9)]) if valid_chars == "default" else valid_chars
    return "".join([choice(chars) for _ in range(length)])


def copy_func(f):
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def correct(word, words):
    max_ = (None, 0)
    for w in words:
        sm = SequenceMatcher(None, word, w)
        if sm.ratio() > max_[1]:
            max_ = (w, sm.ratio())
    return max_


def rget(url):
    return requests.get(url).text


def test(text="", mu=500, sigma=10):
    pritn(f"{text} {nordis(mu, sigma)}")


def req2dict(url):
    return json.loads(requests.get(url).text)


def rel_heat(t, w):
    return round(1.41 - 1.162 * w + 0.98 * t + 0.0124 * w ** 2 + 0.0185 * w * t)


def cform(str_):
    ret = ""
    for index, char in enumerate(str_):
        if char == ".":
            ret += "\N{BULLET}"
        elif char.isdigit() and index > 0 and str_[index - 1] != "+":
            ret += r"\N{SUBSCRIPT number}".replace("number", int_to_word[int(char)]).encode().decode("unicode escape")
        else:
            ret += char
    return ret


def get_clipboard():
    if Platform.os == "windows":
        return tkinter.Tk().clipboard_get()
    elif Platform.os == "darwin":
        return pd_clipboard_get()


def factorial(num):
    ret = 1
    for i in range(1, num + 1):
        ret *= i
    return ret


def find(iter, cond, default=None):
    return next((x for x in iter if cond(x)), default)


def findi(iter, cond, default=None):
    return next((i for i, x in enumerate(iter) if cond(x)), default)


def flatten(oglist):
    flatlist = []
    for sublist in oglist:
        try:
            if isinstance(sublist, str):
                raise TypeError("argument to flatten() must be a non-string sequence")
            for item in sublist:
                flatlist.append(item)
        except TypeError:
            flatlist.append(sublist)
    return flatlist


def safe_file_name(name, os_=None):
    """ Operating system """
    os_ = os_ if os_ is not None else Platform.os
    """ Initializing forbidden characters / names """
    ret, ext = os.path.splitext(name)
    if os_ == "Windows":
        fc = '<>:"/\\|?*0'
        fw = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    elif os_ == "darwin":
        fc = ":/"
    elif os_ == "linux":
        fc = "/"
    """ Removing invalid words (windows) """
    for w in fw:
        if ret.upper() == w:
            ret = ret.replace(w, "").replace(w.lower(), "")
            break
    """ Removing invalid characters """
    ret = "".join([char for char in ret if char not in fc])
    """ Removing invalid trailing characters """
    ret = ret.rstrip(". ")
    """ Adding the extension back """
    ret += ext
    """ Done """
    return ret


def nordis(mu, sigma, r=None, int_=True, ):
    return (int if int_ else lambda_ret)(getattr(r if r is not None else random, "gauss")(mu, sigma))


def txt2list(path_):
    with open(path_, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def first(itr, val, default=None):
    for i, v in enumerate(itr):
        if callable(val):
            if val(v):
                return i
        else:
            if v == val:
                return i
    return default


def cdict(dict_):
    return {k: v for k, v in dict_.items()}


def c1dl(list_):
    return list_[:]


def c2dl(list_):
    return [elm[:] for elm in list_]


def cdil(list_):
    return [{k: v for k, v in elm.items()} for elm in list_]


def solveq(eq, char="x"):
    default = [side.strip() for side in eq.split("=")]
    sides = default[:]
    num = 0
    while True:
        sides = default[:]
        sides = [side.replace(char, "*" + str(num)) for side in sides]
        if eval(sides[0]) == eval(sides[1]):
            break
        else:
            num += 1
    return num


def dis(p1: tuple, p2: tuple) -> int:
    a = p1[0] - p2[0]
    b = p1[1] - p2[1]
    dis = math.sqrt(a ** 2 + b ** 2)
    return dis


def valtok(dict_, value):
    keys = list(dict_.keys())
    values = list(dict_.values())
    return keys[values.index(value)]


def print_error(e: Exception):
    pritn(f"{type(e).__name__ }: ", *e.args)


def name(obj):
    """ Returns the name of an object, i.e. 1 is 'int' and 'foo' is 'str' """
    return type(obj).__name__


def millis(seconds):
    """ Converts seconds to milliseconds, really nothing interesting here """
    return seconds * 1000


def toperc(part, whole, max_=100):
    """ from two numbers (fraction) to percentage (part=3; whole=6 -> 50(%) """
    return part / whole * max_


def fromperc(perc, whole, max_=100):
    """ from percentage to number (perc=50; whole=120 -> 100) """
    return perc * whole / max_


def clocktime(int_):
    """ Returns a string that represents the clock time version of an integer () (2 -> 02) """
    return "0" + str(int_) if len(str(int)) == 1 else str(int_)


def relval(a, b, val):
    """ Returns the appropiate value based on the weight of the first value, i.e. with a=80, b=120 and val=50, it will return 75 """


def lget(l, i, d=None):
    """ Like a dict's get() method, but with indexes (l = list, i = index, d = default) """
    l[i] if i < len(l) else d if d is not None else None


def roundn(num, base=1):
    """ Returns the rounded value of a number with a base to round to (num=5; base=7 -> 7)"""
    return base * round(float(num) / base)


def floorn(num, base=1):
    """ Returns the floor value of a number with a base to round to (num=86, base=50 -> 50) """
    return num - num % base


def chance(chance_):
    """ Returns True based on chance from a float-friendly scale of 0 to 1, i.e. 0.7 has a higher chance of returning True than 0.3 """
    return random.random() < chance_



def isprivate(str_):
    """ Returns whether a string is a dunder/private attribute (starts and ends with a (sing)(doub)le underscore (dunderscore)) """
    return str_.lstrip("_") != str_ or str_.rstrip("_") != str_


def hmtime():
    """ Returns the current time in this format: f"{hours}:{minutes}" """
    return time.strfime("%I:%M")


def revnum(num):
    """ Returns the reverse of a number, i.e. 1 == -1 and -1 == 1 (0 != -0; 0 == 0) """
    return -num if num > 0 else abs(num)


# decorator functions
def scatter(func, stmt, globs, locs):
    while (placeholder := token()) in inspect.getsource(func):
        do_nothing()
    lines = inspect.getsourcelines(func)[0]
    additions = [placeholder] * len(lines)
    code = list(sum(zip(lines, additions), ()))
    n_placeholders = 0
    for index, line in enumerate(code[:]):
        if line == placeholder:
            try:
                checked_line = code[index + 1]
            except IndexError:
                checked_line = code[index - 1]
            indent = ""
            if index != 0:
                for i, char in enumerate(checked_line):
                    if char == " ":
                        indent += " "
                    else:
                        break
            else:
                indent = ""
            extra_indentations = ("elif", "else", "except", "finally")
            for ind in extra_indentations:
                if checked_line.strip().startswith(ind):
                    indent += "    "
                    break
            code[index] = indent + line + "\n"
            n_placeholders += 1
    perc_inc = 100 / n_placeholders
    code = [line[4:] for line in code]
    for index, line in enumerate(code[:]):
        if line.startswith("@"):
            del code[index]
        else:
            break
    del code[:2]
    code = [line[4:] for line in code]
    # replace placeholders
    for index, line in enumerate(code):
        if line.strip(" \n") == placeholder:
            code[index] = line.replace(placeholder + "\n", stmt + ";print(g.loading_world_perc, am); am += 1" + "\n")
    code = "".join(code)
    code = "am = 0\n" + code
    print(code)

    def wrapper():
        try:
            exec(code, globs, locs)
        except:
            #pritn(code)
            raise

    return wrapper


def merge(*funcs):
    def wrapper():
        for func in funcs:
            func()

    return wrapper


profiler = LineProfiler()
def profile(func):
    def inner(*args, **kwargs):
        profiler.add_function(func)
        profiler.enable_by_count()
        return func(*args, **kwargs)

    return inner


# classes
class DictWithoutException(dict):
    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            return DictWithoutException()

    def __repr__(self):
        return f"DWI({dict(self)})"


class Platform:
    os = platform.system().lower()


class Infinity:
    def __init__(self, val="+"):
        self.val = val
        self.repr = ("" if val == "+" else "-") + INF

    @property
    def pos(self):
        return self.val == "+"

    @property
    def neg(self):
        return self.val == "-"

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr

    def __eq__(self, other):
        if isinstance(other, type(self)) and self.val == other.val:
            return True
        return False

    def __lt__(self, other):
        if self.neg:
            if isinstance(other, type(self)) and other.neg:
                return False
            return True
        return False

    def __le__(self, other):
        if self.neg:
            return True
        return False

    def __gt__(self, other):
        if self.pos:
            if isinstance(other, type(self)) and other.pos:
                return False
            return True
        return False

    def __ge__(self, other):
        if self.pos:
            return True
        return False

    def __add__(self, other):
        if self.pos:
            return type(self)()
        elif isinstance(other, type(self)) and other.pos:
            return 0
        return type(self)("-")

    def __radd__(self, other):
        if self.pos:
            if isinstance(other, type(self)):
                if other.pos:
                    return type(self)()
                return 0
            return type(self)()
        if isinstance(other, type(self)) and other.pos:
            return 0
        return type(self)()

    def __sub__(self, other):
        if self.pos:
            if isinstance(other, type(self)):
                if other.pos:
                    return 0
                return type(self)()
            return type(self)()
        elif isinstance(other, type(self)) and other.neg:
            return 0
        return type(self)("-")

    def __rsub__(self, other):
        if self.pos:
            if isinstance(other, type(self)) and other.pos:
                return 0
            return type(self)("-")
        if isinstance(other, type(self)) and other.pos:
            return type(self)()
        return type(self)("-")


class _Translator:
    def __init__(self, service, lang):
        self.service = service
        self.lang = lang
        self.saved = {}
        self.init(self.lang)

    def init(self, lang):
        self.lang = lang
        if not self.saved.get(self.lang, False):
            self.saved[self.lang] = {}

    def add(self, lang, dict_):
        self.saved = {}
        self.saved[lang] = dict_


class TranslatepyTranslator(_Translator):
    def __init__(self, lang="english"):
        super().__init__(_TpyGoogleTranslate(), lang)

    def __or__(self, other):
        if self.lang == "english":
            return other
        if other not in self.saved[self.lang]:
            t = self.service.translate(other, self.lang).result
            self.saved[self.lang][other] = t
            return t
        else:
            return self.saved[self.lang][other]


class GoogletransTranslator(_Translator):
    def __init__(self, lang="english"):
        super().__init__(_GoogleGoogleTranslate(), lang)

    def __or__(self, other):
        if self.lang == "english":
            return other
        if other not in self.saved[self.lang]:
            t = self.service.translate(other, self.lang).text
            self.saved[self.lang][other] = t
            return t
        else:
            return self.saved[self.lang][other]


class Noise:
    def __init__(self, r=None):
        self.r = r if r is not None else Random(3.14)

    def linear(self, average, length, flatness=0, start=None):
        noise = []
        avg = average
        if start is None:
            noise.append(nordis(avg, 2, self.r))
        else:
            noise.append(start)
        for _ in range(length - 1):
            # bounds
            if noise[-1] == avg - 2:
                noise.append(self.r.choice([noise[-1], noise[-1] + 1] + [noise[-1] for _ in range(flatness)]))
            elif noise[-1] == avg + 2:
                noise.append(self.r.choice([noise[-1] - 1, noise[-1]] + [noise[-1] for _ in range(flatness)]))
            # normal
            else:
                n = [-1, 0, 1] + [0 for _ in range(flatness)]
                noise.append(noise[-1] + self.r.choice(n))
        return noise

    def collatz(self, start_num):
        noise = [start_num]
        while noise[-1] != 1:
            if noise[-1] % 2 == 0:
                noise.append(noise[-1] // 2)
            else:
                noise.append(noise[-1] * 3 + 1)
        return noise


class SmartList(list):
    def matrix(self, dims=None):
        return np.array(self).reshape(*(dims if dims is not None else [int(sqrt(len(self)))] * 2)).tolist()

    def to_string(self):
        return "".join(self)

    def moore(self, index, hl=None, area=(0, 1, 2, 3, 5, 6, 7, 8), return_indexes=False):
        neighbors = []
        HL = int(sqrt(len(self))) if hl is None else hl
        indexes = []
        if 0 in area:
            indexes.append(index - HL - 1)
        if 1 in area:
            indexes.append(index - HL)
        if 2 in area:
            indexes.append(index - HL + 1)
        if 3 in area:
            indexes.append(index - 1)
        if 4 in area:
            indexes.append(index)
        if 5 in area:
            indexes.append(index + 1)
        if 6 in area:
            indexes.append(index + HL - 1)
        if 7 in area:
            indexes.append(index + HL)
        if 8 in area:
            indexes.append(index + HL + 1)
        for i in indexes:
            with suppress(IndexError):
                if i >= 0:
                    neighbors.append(self[i])
        if return_indexes:
            return indexes
        return neighbors

    def von_neumann(self, index, hl=None, return_indexes=False):
        return self.moore(index, hl, (2, 4, 6, 8), return_indexes)

    def get(self, index, default):
        return self[index] if index < len(self) else default

    def extendbeginning(self, itr):
        for val in itr:
            self.insert(0, val)

    def smoothen(self, wall, air, deficiency, overdosis, birth, itr, hl):
        for _ in range(itr):
            cop = SmartList(self)
            for i, x in enumerate(cop):
                neighbors = cop.moore(i, hl)
                corners = 8 - len(neighbors)
                if x == wall:
                    if neighbors.count(wall) + corners < deficiency:
                        self[i] = air
                    elif neighbors.count(wall) + corners > overdosis:
                        self[i] = air
                elif x == air:
                    if neighbors.count(wall) + corners == birth:
                        self[i] = wall

    def find(self, cond, retlist=False):
        return [x for x in self if cond(x)][0 if not retlist else slice(0, len(self))]

    @property
    def mean(self):
        return sum(self) / len(self)

    @property
    def median(self):
        return sorted(self)[len(self) // 2]

    @property
    def mode(self):
        freqs = dict.fromkeys(self, 0)
        for elem in self:
            freqs[elem] += 1
        max_ = max(freqs.values())
        return valtok(freqs, max_)


class SmartOrderedDict:
    def __init__(self, dict_=None, **kwargs):
        dict_ = dict_ if dict_ is not None else {}
        self._keys = []
        self._values = []
        for k, v in dict_.items():
            self._keys.append(k)
            self._values.append(v)
        for k, v in kwargs.items():
            self._keys.append(k)
            self._values.append(v)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._values[self._keys.index(key)]
        elif isinstance(key, int):
            return self._values[key]

    def __setitem__(self, key, value):
        if key in self._keys:
            #raise RuntimeError(f"Object of class {type(self)} cannot modify the nonexistent key '{key}'. To create new keys use the 'insert' method instead.")
            self._values[self._keys.index(key)] = value
        else:
            self.insert(0, key, value)

    def __repr__(self):
        return str({k: v for k, v in zip(self._keys, self._values)})

    def __iter__(self):
        return self.keys()

    def __bool__(self):
        return bool(self._keys)

    def __len__(self):
        return len(self._keys)

    def delval(self, val):
        del self._values[self._keys.index(value)]
        self._keys.remove(value)

    def delindex(self, index):
        del self._keys[index]
        del self._values[index]

    def getindex(self, index, key=True):
        return (self._keys if key else self._values)[index]

    def fromvalue(self, value):
        return self._keys[self._values.index(value)]

    def insert(self, index, key, value):
        self._keys.insert(index, key)
        self._values.insert(index, value)

    def keys(self):
        return iter(self._keys)

    def values(self):
        return iter(self._values)

    def items(self):
        return {k: v for k, v in zip(self._keys, self._values)}.items()


class JavaScriptObject(dict):
    def __getattr__(self, value):
        return self[value]


class PseudoRandomNumberGenerator:
    def _last(self, num):
        return int(str(num)[-1])

    def random(self):
        return int(self._last(time.time()) + self._last(cpu_percent()) / 2) / 10


# context managers
class AttrExs:
    def __init__(self, obj, attr, p):
        self.obj = obj
        self.attr = attr
        self.p = p

    def __enter__(self):
        if not hasattr(self.obj, self.attr):
            setattr(self.obj, self.attr, self.p)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return isinstance(exc_type, Exception)


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class Shut:
    def __enter__(self):
        do_nothing()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class DThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDaemon(True)


# constants
InvalidFilenameError = FileNotFoundError
BreakAllLoops = create_exception("BreakAllLoops", Exception)
ArbitraryException = create_exception("ArbitraryException", Exception)
