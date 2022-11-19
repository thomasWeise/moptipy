"""Test all the links in the project's README.md file."""
import os.path
from time import sleep
from typing import Final

# noinspection PyPackageRequirements
import certifi

# noinspection PyPackageRequirements
import urllib3

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.strings import replace_all


def __ve(msg: str, text: str, idx: int) -> ValueError:
    """
    Raise a value error for the given text piece.

    :param msg: the message
    :param text: the string
    :param idx: the index
    :returns: a value error ready to be raised
    """
    piece = text[max(0, idx - 32):min(len(text), idx + 64)].strip()
    return ValueError(f"{msg}: '...{piece}...'")


#: the headers
__HEADER: Final[dict[str, str]] = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64;"
                  " rv:106.0) Gecko/20100101 Firefox/106.0"
}


def __needs_body(base_url: str) -> bool:
    """
    Check whether we need the body of the given url.

    :param base_url: the url string
    :returns: `True` if the body is needed, `False` otherwise
    """
    if base_url.endswith(".html") or base_url.endswith(".htm"):
        return True
    return base_url.startswith("https://yaml.org/spec/")


def __check(url: str, valid_urls: dict[str, str | None],
            http: urllib3.PoolManager = urllib3.PoolManager(
                cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())) -> None:
    """
    Check if a url exists.

    :param url: str
    :param valid_urls: the set of valid urls
    :param http: the pool manager
    """
    if (url != url.strip()) or (len(url) < 3):
        raise ValueError(f"invalid url '{url}'")
    if url in valid_urls:
        return
    if url.startswith("mailto:"):
        return
    if not url.startswith("http"):
        raise ValueError(f"invalid url '{url}'")

    base_url: str = url
    selector: str | None = None
    needs_body: bool
    i = url.find("#")
    if i >= 0:
        base_url = url[:i]
        needs_body = __needs_body(base_url)
        if not needs_body:
            raise ValueError(f"invalid url: '{url}'")

        selector = url[i + 1:]
        if (len(selector) <= 0) or (len(base_url) <= 0) \
                or len(selector.strip()) != len(selector) \
                or (len(base_url.strip()) != len(base_url)):
            raise ValueError(f"invalid url: '{url}'")

        if base_url in valid_urls:
            body = valid_urls[base_url]
            if body is None:
                raise ValueError(
                    f"no body for '{url}' with base '{base_url}'??")
            for qt in ("", "'", '"'):
                if f"id={qt}{selector}{qt}" in body:
                    return
            raise ValueError(
                f"did not find id='{selector}' of '{url}' in body "
                f"of '{base_url}': '{body}'")
    else:
        needs_body = __needs_body(base_url)

    code: int
    body: str | None
    method = "GET" if needs_body else "HEAD"
    try:
        sleep(0.5)
        response = http.request(method, base_url, timeout=20, redirect=True,
                                retries=5, headers=__HEADER)
        code = response.status
        body = response.data.decode("utf-8") if needs_body else None
    except BaseException as be:
        # sometimes, I cannot reach github from here...
        if url.startswith("http://github.com") \
                or url.startswith("https://github.com"):
            return
        raise ValueError(f"invalid url '{url}'.") from be

    logger(f"checked url '{url}' got code {code} for method '{method}' and "
           f"{0 if body is None else len(body)} chars.")
    if code != 200:
        raise ValueError(f"url '{url}' returns code {code}.")

    if selector is not None:
        for qt in ("", "'", '"'):
            if f"id={qt}{selector}{qt}" in body:
                return
        raise ValueError(
            f"did not find id='{selector}' of '{url}' in body "
            f"of '{base_url}': '{body}'")

    if needs_body and (body is None):
        raise ValueError(f"huh? body for '{url}' / '{base_url}' is None?")

    valid_urls[base_url] = body
    if url != base_url:
        valid_urls[url] = None


def test_all_links_in_readme_md() -> None:
    """Test all the links in the README.md file."""
    # First, we load the README.md file as a single string
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    readme = Path.file(base_dir.resolve_inside("README.md"))
    logger(f"testing all links from README.md file '{readme}'.")
    text = readme.read_all_str()
    logger(f"got {len(text)} characters.")
    if len(text) <= 0:
        raise ValueError(f"README.md file at '{readme}' is empty?")
    del readme

    # remove all code blocks
    start: int = -1
    lines: Final[list[str]] = []
    while True:
        start += 1
        i = text.find("\n```", start)
        if i < start:
            lines.append(text[start:].strip())
            break
        j = text.find("\n```", i + 1)
        if j < i:
            raise __ve("Multi-line code start without end", text, i)
        k = text.find("\n", j + 1)
        if k < j:
            raise __ve("Code end without newline", text, i)
        lines.append(text[start:i].strip())
        start = k

    text = "\n".join(lines).strip()
    lines.clear()

    # these are all urls that have been verified
    valid_urls: Final[dict[str, str | None]] = {}

    # build the map of local reference marks
    start = -1
    while True:
        start += 1
        i = text.find("\n#", start)
        if i < start:
            break
        j = text.find(" ", i + 1)
        if j < i:
            raise __ve("Headline without space after #", text, i)
        k = text.find("\n", j + 1)
        if k < j:
            raise __ve("Headline without end", text, i)
        rid: str = text[j:k].strip().replace(" ", "-")
        for ch in ".:,()`/":
            rid = rid.replace(ch, "")
        rid = replace_all("--", "-", rid).lower()
        if (len(rid) <= 2) or ((rid[0] not in "123456789")
                               and (start > 0)) or ("-" not in rid):
            raise __ve(f"invalid id '{rid}'", text, i)
        valid_urls[f"#{rid}"] = None
        start = k

    # remove all inline code
    start = -1
    while True:
        start += 1
        i = text.find("`", start)
        if i < start:
            lines.append(text[start:].strip())
            break
        j = text.find("`", i + 1)
        if j < i:
            raise __ve("Multi-line code start without end", text, i)
        lines.append(text[start:i].strip())
        start = j
    text = "\n".join(lines).strip()
    lines.clear()

    logger("now checking '![...]()' style urls")

    # now gather the links to images and remove them
    start = -1
    lines.clear()
    while True:
        start += 1
        i = text.find("![", start)
        if i < start:
            lines.append(text[start:])
            break
        j = text.find("]", i + 1)
        if j <= i:
            break
        if "\n" in text[i:j]:
            start = i
        j += 1
        if text[j] != "(":
            raise __ve("invalid image sequence", text, i)
        k = text.find(")", j + 1)
        if k <= j:
            raise __ve("no closing gap for image sequence", text, i)

        __check(text[j + 1:k], valid_urls)

        lines.append(text[start:i])
        start = k

    text = "\n".join(lines)
    lines.clear()

    logger("now checking '[...]()' style urls")

    # now gather the links and remove them
    start = -1
    lines.clear()
    while True:
        start += 1
        i = text.find("[", start)
        if i < start:
            lines.append(text[start:])
            break
        j = text.find("]", i + 1)
        if j <= i:
            break
        if "\n" in text[i:j]:
            lines.append(text[start:i])
            start = i
            continue
        j += 1
        if text[j] != "(":
            raise __ve("invalid link", text, i)
        k = text.find(")", j + 1)
        if k <= j:
            raise __ve("no closing gap for link", text, i)

        __check(text[j + 1:k], valid_urls)

        lines.append(text[start:i])
        start = k

    text = "\n".join(lines)
    lines.clear()

    logger("now checking ' href=' style urls")

    # now gather the href links and remove them
    for quot in "'\"":
        start = -1
        lines.clear()
        while True:
            start += 1
            start_str = f" href={quot}"
            i = text.find(start_str, start)
            if i < start:
                lines.append(text[start:])
                break
            j = text.find(quot, i + len(start_str))
            if j <= i:
                break
            if "\n" in text[i:j]:
                lines.append(text[start:i])
                start = i
                continue
            __check(text[i + len(start_str):j], valid_urls)

            lines.append(text[start:i])
            start = j

        text = "\n".join(lines)
        lines.clear()

    logger("now checking ' src=' style urls")
    # now gather the image links and remove them
    for quot in "'\"":
        start = -1
        lines.clear()
        while True:
            start += 1
            start_str = f" src={quot}"
            i = text.find(start_str, start)
            if i < start:
                lines.append(text[start:])
                break
            j = text.find(quot, i + len(start_str))
            if j <= i:
                break
            if "\n" in text[i:j]:
                lines.append(text[start:i])
                start = i
                continue
            __check(text[i + len(start_str):j], valid_urls)

            lines.append(text[start:i])
            start = j

        text = "\n".join(lines)
        lines.clear()

    logger("finished testing all links from README.md.")
