from .imports import *
import operator as op
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
from line_profiler import LineProfiler


# constants
char_mods = {"a": "áàâãä",
             "e": "éèêë",
             "i": "íìîï",
             "o": "óòôõö",
             "u": "úùûü"}
phi = (1 + sqrt(5)) / 2
int_to_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# take a moment to appeciate this beuaty
print = print
pritn = print
prrint = print
pirnt = print
PRINT = print 
pint = print 
ptrint = print
priint = print
prinit = print
prinnt = print
priinit = print
prnit = print
print = print   # I'm gonna move to a rural farm to be a blacksmith for the rest of my life if I mispell print in yet anoother different fashon I stg
print = print

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
def delay(func, secs, *args, **kwargs):
    def inner():
        sleep(secs)
        func(*args, **kwargs)

    Thread(target=inner).start()


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


def audio_length(p):
    with wave.open(p, "r") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / rate
        return duration


def is_normal(s):
    for char in s:
        if char not in string.ascii_lowercase + string.ascii_uppercase + string.punctuation + string.digits + "".join(char_mods.values()) + " ":
            return False
    return True


def shake_str(s):
    return bubble_sort(s, lambda x, y: chance(1 / 4), 1)


def raise_type(argname, argindex, right, wrong):
    raise TypeError(f"Argument {argindex} ({argname}) must be of type '{right}', not '{wrong}'")


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


def do_nothing():
    pass


def open_text(filepath):
    if Platform.os == "windows":
        os.system(f"notepad {filepath}")
    elif Platform.os == "darwin":
        os.system(f"open {filepath}")


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


def test(text="", mu=500, sigma=10):
    pritn(f"{text} {nordis(mu, sigma)}")


def req2dict(url):
    return json.loads(requests.get(url).text)


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


def nordis(mu, sigma, int_=True):
    f = (int if int_ else lambda_ret)
    ret = f(random.gauss(mu, sigma))
    ret = min(ret, mu + sigma)
    ret = max(ret, mu - sigma)
    return ret


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


def valtok(dict_, value):
    keys = list(dict_.keys())
    values = list(dict_.values())
    return keys[values.index(value)]


def print_error(e: Exception):
    pritn(f"{type(e).__name__ }: ", *e.args)


def clamp(value, min_, max_):
    return min(max(value, min_), max_)


def millis(seconds):
    """ Converts seconds to milliseconds, really nothing interesting here """
    return seconds * 1000


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
    """ Returns whether a string is a dunder/private attribute (starts and ends with a (sing)(doub)le underscore (dunderscore) (you~re acoustic)) """
    return str_.lstrip("_") != str_ or str_.rstrip("_") != str_


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
class Symbols:
    INF = "\u221e"  # infinity
    DEG = "\u00B0"  # celcius
    BULLET = "⁍"  # bullet
    TM = chr(8482)


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
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True


class DThread(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon = True


# constants
InvalidFilenameError = FileNotFoundError
BreakAllLoops = create_exception("BreakAllLoops", Exception)
ArbitraryException = create_exception("ArbitraryException", Exception)
