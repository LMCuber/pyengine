from .imports import *
from translatepy.translators.google import GoogleTranslate as _TpyGoogleTranslate
from googletrans import Translator as _GoogleGoogleTranslate
import numpy as np
import inspect
import tkinter
import sys
import os
import io
import random
import platform
import time
import pycountry
import requests
import string
import json
import inspect


# constants
INF = "\u221e"  # infinity symbol (unicode)
int_to_word = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
pritn = print  # because I always mess it up when typing fast
prrint = print # because I always mess it up when typing fast


# functions
def choose_file(title, file_types="all"):
    kw = {"filetypes": file_types} if file_types != "all" else {}
    if Platform.os != "Darwin":
        p = tkinter.filedialog.askopenfilename(title=title, **kw)
        return p
    else:
        return NotImplemented


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
    

def notepadopen(filepath):
    os.system(f"notepad {filepath}")
    

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


def test():
    print(rand(0, 999))
    
    
def req2dict(url):
    return json.loads(requests.get(url).text)


def rel_heat(t, w):
    return round(1.41 - 1.162 * w + 0.98 * t + 0.0124 * w ** 2 + 0.0185 * w * t)


def iso_countries():
    return [country.alpha_2 for country in pycountry.countries]


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
    return tkinter.Tk().clipboard_get()
    
    
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
    
    
def safe_file_name(name):
    """ Initializing forbidden characters / names """
    ret, ext = os.path.splitext(name)
    if Platform.os == "Windows":
        fc = '<>:"/\\|?*0'
        fw = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
    elif Platform.os == "Darwin":
        fc = ":/"
    elif Platform.os == "Linux":
        fc = "/"
    """ Removing invalid words (Windows) """
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
    

def nordis(mu, sigma):
    return int(random.gauss(mu, sigma))


def txt2list(path_):
    with open(path_, "r") as f:
        return [line.rstrip("\n") for line in f.readlines()]


def first(itr, val, p):
    for i, v in enumerate(itr):
        if v == val:
            return i
    return p


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
    print(type(e).__name__ + ": ", *e.args)
    
    

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
            code[index] = line.replace(placeholder + "\n", stmt + "\n")
    code = "".join(code)
        
    def wrapper():
        exec(code, globs, locs)
    
    return wrapper


def merge(*funcs):
    def wrapper():
        for func in funcs:
            func()
            
    return wrapper


def delay(secs):
    def decorator(func):
        def wrapper():
            Thread(target=thread).start()

        def thread():
            time.sleep(secs)
            func()

        return wrapper

    return decorator
    

# classes
class Platform:
    os = platform.system()


BreakAllLoops = create_exception("BreakAllLoops", Exception)
ArbitraryException = create_exception("ArbitraryException", Exception)


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
        self.saved = {}
        self.init(self.lang)
    
    def init(self, lang):
        self.lang = lang
        if not self.saved.get(self.lang, False):
            self.saved[self.lang] = {}
        

class TranslatepyTranslator(_Translator):
    def __init__(self, lang="english"):
        super().__init__(_TpyGoogleTranslate(), lang)
    
    def __or__(self, other):  # | operator (bitwise or)
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
    
    def __or__(self, other):  # | operator (bitwise or)
        if self.lang == "english":
            return other
        if other not in self.saved[self.lang]:
            t = self.service.translate(other, self.lang).text
            self.saved[self.lang][other] = t
            return t
        else:
            return self.saved[self.lang][other]
               
               
class Noise:
    @staticmethod
    def linear(average, length, flatness=0):
        noise = []
        avg = average
        noise.append(nordis(avg, 2))
        for _ in range(length - 1):
            # bounds
            if noise[-1] == avg - 2:
                noise.append(choice([noise[-1], noise[-1] + 1] + [noise[-1] for _ in range(flatness)]))
            elif noise[-1] == avg + 2:
                noise.append(choice([noise[-1] - 1, noise[-1]] + [noise[-1] for _ in range(flatness)]))
            # normal
            else:
                n = [-1, 0, 1] + [0 for _ in range(flatness)]
                noise.append(noise[-1] + choice(n))
        return noise
    
    @staticmethod
    def collatz(start_num):
        noise = [start_num]
        while noise[-1] != 1:
            if noise[-1] % 2 == 0:
                noise.append(noise[-1] / 2)
            else:
                noise.append(noise[-1] * 3 + 1)
        return noise
        
        
class SmartList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def matrix(self, dims=None):
        return np.array(self).reshape(*(dims if dims is not None else [int(sqrt(len(self)))] * 2)).tolist()
    
    def to_string(self):
        return "".join(self)
    
    def moore(self, index, hl=None, area="edgescorners"):
        neighbors = []
        HL = int(sqrt(len(self))) if hl is None else hl
        indexes = []
        if "edges" in area:
            indexes.extend((                index - HL,
                            index - 1,                  index + 1,
                                            index + HL,           ))
        if "corners" in area:
            indexes.extend((index - HL - 1,             index + HL + 1,
                            
                            index + HL - 1,             index + HL + 1))
        for i in indexes:
            with suppress(IndexError):
                neighbors.append(self[i])
        return neighbors
        
    def get(self, index, default):
        return self[index] if index < len(self) else default
    
    def extendbeginning(self, itr):
        for val in itr:
            self.insert(0, val)
        
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
        for elem in freqs:
            if freqs[elem] == max_:
                return elem
                
                
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
    
    def delval(self, val):
        del self._values[self._keys.index(value)]
        self._keys.remove(value)
    
    def delindex(self, index):
        del self._keys[index]
        del self._values[index]
        
    def fromvalue(self, value):
        return self._keys[self._values.index(value)]
    
    def insert(self, index, key, value):
        self._keys.insert(index, key)
        self._values.insert(index, value)
    
    def keys(self):
        return iter(self._keys)
    
    def values(self):
        return iter(self._values)
        
        
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
        

# constants
InvalidFilenameError = FileNotFoundError
