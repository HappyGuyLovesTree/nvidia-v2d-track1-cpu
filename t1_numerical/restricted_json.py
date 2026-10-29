"""Extract explicit JSON pointers without materialising other JSON values.

The immutable full bytes may be hashed. Unselected values are lexically skipped;
they are never passed to json.loads/raw_decode. '*' denotes an array index.
"""
import hashlib
import json
import re
from pathlib import Path

NUMBER = re.compile(r'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_ref(path):
    p = Path(path).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=h.hexdigest())


def legacy_content_digest(path):
    """Hash original sorted compact JSON under tracking.digest's ASCII encoding.

    No values are deserialised, including skipped prediction paths/values. This
    encoding-only transform is valid for the original sorted compact writer.
    """
    text = Path(path).read_text(encoding='utf-8')
    quoted = escaped = False
    chunks = []
    for ch in text:
        if quoted:
            chunks.append(ch if ord(ch) < 128 else json.dumps(ch)[1:-1])
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
            chunks.append(ch)
        elif ch not in ' \t\r\n':
            chunks.append(ch)
    if quoted:
        raise ValueError('unterminated encoding input')
    return hashlib.sha256(''.join(chunks).encode('ascii')).hexdigest()


def extract(text, pointers):
    trie = {}
    for pointer in pointers:
        node = trie
        for part in pointer.strip('/').split('/'):
            node = node.setdefault(part, {})
        node['$'] = True
    decoder = json.JSONDecoder()
    n = len(text)

    def ws(i):
        while i < n and text[i] in ' \t\r\n':
            i += 1
        return i

    def string_end(i):
        if text[i] != '"':
            raise ValueError('expected JSON string')
        i += 1
        while i < n:
            if text[i] == '"':
                return i + 1
            if text[i] == '\\':
                i += 2
            else:
                i += 1
        raise ValueError('unterminated JSON string')

    def skip(i):
        i = ws(i)
        if i >= n:
            raise ValueError('truncated JSON')
        c = text[i]
        if c == '"':
            return string_end(i)
        if c in '[{':
            closing = ']' if c == '[' else '}'
            stack = [closing]
            i += 1
            while stack and i < n:
                c = text[i]
                if c == '"':
                    i = string_end(i)
                    continue
                if c in '[{':
                    stack.append(']' if c == '[' else '}')
                elif c in ']}':
                    if c != stack.pop():
                        raise ValueError('unbalanced JSON')
                i += 1
            if stack:
                raise ValueError('truncated JSON container')
            return i
        for literal in ('true', 'false', 'null'):
            if text.startswith(literal, i):
                return i + len(literal)
        match = NUMBER.match(text, i)
        if not match:
            raise ValueError('invalid JSON scalar')
        return match.end()

    def parse(i, node):
        i = ws(i)
        if '$' in node:
            return decoder.raw_decode(text, i)
        c = text[i]
        if c == '{':
            out, seen = {}, set()
            i = ws(i + 1)
            while text[i] != '}':
                end = string_end(i)
                key = json.loads(text[i:end])
                if key in seen:
                    raise ValueError('duplicate JSON key')
                seen.add(key)
                i = ws(end)
                if text[i] != ':':
                    raise ValueError('expected colon')
                i = ws(i + 1)
                if key in node:
                    out[key], i = parse(i, node[key])
                else:
                    i = skip(i)
                i = ws(i)
                if text[i] == '}':
                    break
                if text[i] != ',':
                    raise ValueError('expected comma')
                i = ws(i + 1)
            return out, i + 1
        if c == '[' and '*' in node:
            out, i = [], ws(i + 1)
            while text[i] != ']':
                value, i = parse(i, node['*'])
                out.append(value)
                i = ws(i)
                if text[i] == ']':
                    break
                if text[i] != ',':
                    raise ValueError('expected array comma')
                i = ws(i + 1)
            return out, i + 1
        raise ValueError('pointer container missing or wrong type')

    result, end = parse(0, trie)
    if ws(end) != n:
        raise ValueError('trailing JSON')
    return result


def read_selected(entry):
    p = Path(entry['path'])
    data = p.read_bytes()
    if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
        raise ValueError('frozen read reference changed: ' + str(p))
    return extract(data.decode('utf-8'), entry['pointers'])


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    return file_ref(path)
