from html.parser import HTMLParser
from html import unescape
from collections import namedtuple, deque
import re

Clause = namedtuple('Clause', ('title', 'body'))
taggedData = namedtuple('taggedData', ('data', 'tag', 'index'))

predefined_tokens = {'number': "{num}", 'enum': "{enum}", 'pad': "{pad}", 'unknown': "{unk}"}
not_alpha_pattern = re.compile(r'[^A-Za-z\s#]+')
not_alpha_pattern_punct = re.compile(r'[^A-Za-z\s#\,\.\!\?\(\)]+')
numeric_pattern = re.compile('\d[\d\.,]+')
enum_pattern = re.compile('^(?:\d+\.\s)|^(?:\(\d+\)\s)|^(?:[a-zA-Z]+\.\s)|^(?:\([a-zA-Z]+\)\s)')
upper_pattern = re.compile(r"[A-Z]+[\s\.,:';\(\)A-Z]+[\n\s]]")


def tokenizer(txt, lower=True, enum=False, numeric=True, split=True):
    # preprocessing
    if lower:
        txt = txt.lower()
    if enum:
        txt = enum_pattern.sub(predefined_tokens["enum"], txt)
    if numeric:
        txt = numeric_pattern.sub(predefined_tokens["number"], txt)
    # splitting
    if not split:
        return txt
    for c in '()[]./,;:"':
        txt = txt.replace(c, f" {c} ")
    return [w for w in txt.split() if any(w)]


def to_float(s):
    """Forces a conversion to float"""
    try:
        return float("".join([c for c in s if s.isdigit() or c == '.']))
    except:
        return 0.0


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


class StyleExtrater(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.stack = deque()
        self.indices = deque()
        self.indices.append(0)
        self.tagged = []

    def starttag_of_interest(self, tag, styles):
        tag = tag.lower()
        if tag in ['t']:
            return 't'
        if tag in ['b', 'strong']:
            return 'b'
        if tag in ['i', 'em']:
            return 'i'
        if tag in ['u']:
            return 'u'
        if tag in ['h1', 'h2']:  # ,'h3','h4','h5','h6']:
            return 'h'
        if 'font-size' in styles:
            font_size = 10.0
            if styles['font-size'].find('%') > -1:
                font_size *= to_float(styles['font-size'])
            elif styles['font-size'].replace('pt', 'px').find('px') > -1:
                font_size = to_float(styles['font-size'])
                if styles['font-size'].find("pt") >= 0:
                    font_size /= 0.75  # pt to px conversion
            if font_size >= 200:
                return ('h', tag)
        if styles.get('text-decoration') == 'underline':
            return ('u', tag)
        if styles.get('font-style') == 'italic':
            return ('i', tag)
        if styles.get('font-weight') == 'bold' or (
                type(styles.get('font-weight', 0)) == int and (styles.get('font-weight', 0) >= 500)):
            return ('b', tag)
        return None

    def endtag_of_interest(self, tag):
        if not any(self.stack):
            return False
        tag = tag.lower()
        if type(self.stack[-1]) == tuple:  # complex tag
            if tag == self.stack[-1][1]:
                return True
        else:
            if tag == 't' and self.stack[-1] == 't':
                return True
            if tag in ['b', 'strong'] and self.stack[-1] == 'b':
                return True
            if tag in ['i', 'em'] and self.stack[-1] == 'i':
                return True
            if tag in ['u'] and self.stack[-1] == 'u':
                return True
            if self.stack[-1] == 'h' and tag in ['h1', 'h2']:  # ,'h3','h4','h5','h6']:
                return True

        return False

    def parse_style_tag(self, style_tag):
        """Returns an array of tuples defining the style"""
        return [tuple(map(lambda s: s.strip(), styl.split(':', 1))) for styl in style_tag.split(';')]

    def parse_class_tag(self, class_tag):
        """Should get a class tag and return the derived styles"""
        classes = class_tag.split()
        # Not implemented
        return []

    def handle_starttag(self, tag, attrs):
        attrs = dict([(k.lower(), v.lower()) for k, v in attrs if type(k) == type(v) == str])
        styles = []
        if 'style' in attrs:
            styles.extend(self.parse_style_tag(attrs['style']))
        if 'class' in attrs:
            styles.extend(self.parse_class_tag(attrs['class']))
        if 'align' in attrs:
            styles.append(("text-align", attrs["align"]))
        if tag.lower() in ['div', 'p', 'tr', 'li']:
            self.indices.append(0)
        elif tag.lower() in ['br', 'hr']:
            self.indices[-1] = 0
        styles = dict(filter(lambda t: len(t) == 2, styles))
        tag = self.starttag_of_interest(tag, styles)
        if tag is None:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.endtag_of_interest(tag):
            self.stack.pop()
        elif tag.lower() in ['div', 'p', 'tr', 'li']:
            if len(self.indices) > 1:
                self.indices.pop()
            else:
                self.indices[-1] = 0

    def handle_data(self, d):
        if not any(self.stack):
            tag = 'n'
        elif type(self.stack[-1]) == tuple:  # complex tag
            tag = self.stack[-1][0]
        else:  # Simple tag
            tag = self.stack[-1]
        d = d.strip()
        if any(d):
            self.tagged.append(taggedData(d, tag, self.indices[-1]))
            self.indices[-1] += 1

    def get_data(self):
        return self.tagged

    def get_lines(self):
        arr = []
        for t in self.tagged:
            if t.index == 0:
                if any(arr):
                    arr[0] = taggedData(enum_pattern.sub(predefined_tokens["enum"] + ' ', arr[0].data + ' '),
                                        arr[0].tag, 0)
                    yield arr
                arr = []
            arr.append(t)
        if any(arr):
            arr[0] = taggedData(enum_pattern.sub(predefined_tokens["enum"] + ' ', arr[0].data + ' '),
                                arr[0].tag, 0)
            yield arr



def uppercase2Ttag(txt):
    return upper_pattern.sub(lambda x: "<t>"+x.group(0).title()+"</t>" if len(x.group(0).strip().split())>1 else x.group(0), txt)


def strip_tags(fname):
    with open(fname, 'r') as f:
        html = f.read()
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def parse_lines(fname, tokenize=True):
    styler = StyleExtrater()
    if fname.find("<") >= 0:
        html = fname
    else:
        char_dict = {
            ord('*'): None,
            ord('\\'): None,
            ord('['): None,
            ord(']'): None,
            ord('`'): ord("'"),
            ord('’'): ord("'"),
            8220: ord('"'),
            8221: ord('"'),
            160: ord(' '),
        }
        with open(fname, 'rb') as f:
            html = f.read().decode('utf8', errors='ignore')
        html = uppercase2Ttag(unescape(html).translate(char_dict))
    styler.feed(html)
    if not tokenize:
        return list(styler.get_lines())
    ret = []
    for line in styler.get_lines():
        new_line = []
        i=0
        for td in line:
            for token in tokenizer(td.data):
                new_line.append(taggedData(token, td.tag, i))
                i+=1
        ret.append(new_line)
    return ret

if __name__ == "__main__":
    from sys import argv
    from glob import glob
    out_file, in_files = (argv[1], argv[0])
    with open(out_file, 'w') as f:
        for fname in glob(in_files):
            for line in parse_lines(fname):
                for td in line:
                    d = ''.join(c for c in td.data if 0 < ord(c) < 127)
                    f.write(f"{d} {td.tag}\n")
                f.write('\n')
