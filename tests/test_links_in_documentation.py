"""Test all the links in the project's *.md files."""
import os.path
from os import environ
from random import choice
from time import sleep
from typing import Final

# noinspection PyPackageRequirements
import certifi

# noinspection PyPackageRequirements
import urllib3

# noinspection PyPackageRequirements
from urllib3.util.url import Url, parse_url

from moptipy.utils.console import logger
from moptipy.utils.path import Path
from moptipy.utils.strings import replace_all

#: The hosts that somtimes are unreachable from my local machine.
#: When the test is executed in a GitHub workflow, all hosts should be
#: reachable.
SOMETIMES_UNREACHABLE_HOSTS: Final[set[str]] = \
    () if "GITHUB_JOB" in environ else \
    {"github.com", "img.shields.io", "pypi.org", "docs.python.org"}


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


#: The headers to use for the HTTP requests.
#: It seems that some websites may throttle requests.
#: Maybe by using different headers, we can escape this.
__HEADERS: Final[tuple[dict[str, str], ...]] = tuple([
    {"User-Agent": ua} for ua in [
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:106.0) Gecko/20100101"
        " Firefox/106.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like "
        "Gecko) Chrome/109.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, "
        "like Gecko) Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.55",
        "Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 "
        "Version/12.16.2",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; AS; rv:11.0) "
        "like Gecko",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14"
        " (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A",
        "Mozilla/5.0 (PLAYSTATION 3; 3.55)",
    ]])


def __needs_body(base_url: str) -> bool:
    """
    Check whether we need the body of the given url.

    If the complete body of the document needs to be downloaded, this function
    returns `True`. This is the case, for example, if we are talking about
    html documents. In this case, we need to (later) scan for internal
    references, i.e., for stuff like `id="..."` attributes. However, if the
    url does not point to an HTML document, maybe a PDF, then we do not need
    the whole body and return `False`. In the latter case, it is sufficient to
    do a `HEAD` HTTP request, in the former case we need a full `GET`.

    :param base_url: the url string
    :returns: `True` if the body is needed, `False` otherwise
    """
    return base_url.endswith((".html", ".htm", "/"))


def __check(url: str, valid_urls: dict[str, str | None],
            http: urllib3.PoolManager = urllib3.PoolManager(
                cert_reqs="CERT_REQUIRED", ca_certs=certifi.where())) -> None:
    """
    Check if an url can be reached.

    :param url: str
    :param valid_urls: the set of valid urls
    :param http: the pool manager
    """
    if (url != url.strip()) or (len(url) < 3):
        raise ValueError(f"invalid url {url!r}")
    if url in valid_urls:
        return
    if url.startswith("mailto:"):
        return
    if not url.startswith("http"):
        raise ValueError(f"invalid url {url!r}")

    base_url: str = url
    selector: str | None = None
    needs_body: bool
    i = url.find("#")
    if i >= 0:
        base_url = url[:i]
        needs_body = __needs_body(base_url)
        if not needs_body:
            raise ValueError(f"invalid url: {url!r}")

        selector = url[i + 1:]
        if (len(selector) <= 0) or (len(base_url) <= 0) \
                or len(selector.strip()) != len(selector) \
                or (len(base_url.strip()) != len(base_url)):
            raise ValueError(f"invalid url: {url!r}")

        if base_url in valid_urls:
            body = valid_urls[base_url]
            if body is None:
                raise ValueError(
                    f"no body for {url!r} with base {base_url!r}??")
            for qt in ("", "'", '"'):
                if f"id={qt}{selector}{qt}" in body:
                    return
            raise ValueError(
                f"did not find id={selector!r} of {url!r} in body "
                f"of {base_url!r}: {body!r}")
    else:
        needs_body = __needs_body(base_url)

    code: int
    body: str | None
    method = "GET" if needs_body else "HEAD"
    error: Exception | None = None
    response = None
# Sometimes, access to the URLs on GitHub fails.
# I think they probably throttle access from here.
# Therefore, we first do a request with 5s timeout and 0 retries.
# If that fails, we wait 2 seconds and try with timeout 8 and 3 retries.
# If that fails, we wait for 5s, then try with timeout 30 and 3 retries.
# If that fails too, we assume that the URL is really incorrect, which rarely
# should not be the case (justifying the many retries).
    try:
        for sltrt in [(0, 0, 5), (2, 3, 8), (5, 3, 30)]:
            sleep_time, retries, timeout = sltrt
            if sleep_time > 0:
                sleep(sleep_time)
            header: dict[str, str] = choice(__HEADERS)
            try:
                response = http.request(
                    method, base_url, timeout=timeout, redirect=True,
                    retries=retries, headers=header)
                error = None
                break
            except Exception as be:
                logger(f"sleep={sleep_time}, retries={retries}, "
                       f"timeout={timeout}, error={str(be)!r}, and "
                       f"header={header!r} for {base_url!r}.")
                if error is not None:
                    bz = be
                    while True:
                        if bz.__cause__ is None:
                            bz.__cause__ = error
                            break
                        bz = bz.__cause__
                error = be
        if error is not None:
            raise error  # noqa
        if response is None:
            raise ValueError(f"no response from url={base_url!r}?")  # noqa
        code = response.status
        body = response.data.decode("utf-8") if needs_body else None
    except Exception as be:
        # sometimes, I cannot reach github from here...
        parsed: Final[Url] = parse_url(url)
        host: Final[str | None] = parsed.hostname
        if host is None:
            raise ValueError(f"url {url!r} has None as host??") from be
        if host in SOMETIMES_UNREACHABLE_HOSTS:  # sometimes not reachable!
            return
        raise ValueError(f"invalid url {url!r}.") from be

    logger(f"checked url {url!r} got code {code} for method {method!r} and "
           f"{0 if body is None else len(body)} chars.")
    if code != 200:
        raise ValueError(f"url {url!r} returns code {code}.")

    if selector is not None:
        for qt in ("", "'", '"'):
            if f"id={qt}{selector}{qt}" in body:
                return
        raise ValueError(
            f"did not find id={selector!r} of {url!r} in body "
            f"of {base_url!r}: {body!r}")

    if needs_body and (body is None):
        raise ValueError(f"huh? body for {url!r} / {base_url!r} is None?")

    valid_urls[base_url] = body
    if url != base_url:
        valid_urls[url] = None


def check_links_in_file(file: str) -> None:
    """
    Test all the links in the README.md file.

    :param file: the file to check
    """
    # First, we load the file as a single string
    base_dir = Path.directory(os.path.join(os.path.dirname(__file__), "../"))
    readme = Path.file(base_dir.resolve_inside(file))
    logger(f"testing all links from the {file!r} file {readme!r}.")
    text = readme.read_all_str()
    logger(f"got {len(text)} characters.")
    if len(text) <= 0:
        raise ValueError(f"{file!r} file at {readme!r} is empty?")
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
            raise __ve(f"invalid id {rid!r}", text, i)
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

    # now checking <...>-style URLs
    start = -1
    lines.clear()
    while True:
        start += 1
        i = text.find("<http", start)
        if i < start:
            lines.append(text[start:])
            break
        j = text.find(">", i + 1)
        if j <= i:
            break
        if "\n" in text[i:j]:
            lines.append(text[start:i])
            start = i
            continue
        __check(text[i + 1:j], valid_urls)

        lines.append(text[start:i])
        start = j

    logger(f"finished testing all links from {file!r}.")


def test_all_links_in_readme_md() -> None:
    """Test all the links in the README.md file."""
    check_links_in_file("README.md")


def test_all_links_in_contributing_md() -> None:
    """Test all the links in the CONTRIBUTING.md file."""
    check_links_in_file("CONTRIBUTING.md")


def test_all_links_in_security_md() -> None:
    """Test all the links in the SECURITY.md file."""
    check_links_in_file("SECURITY.md")


def test_all_links_in_license() -> None:
    """Test all the links in the LICENSE file."""
    check_links_in_file("LICENSE")
