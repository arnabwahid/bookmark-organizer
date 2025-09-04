#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize Chrome/Firefox bookmarks exported as bookmarks.html (Netscape format).

Zero-parameter script:
  • Expects `bookmarks.html` in the same directory as this script
  • Writes `bookmarks_sorted.html` in the same directory

What it does:
  • Deletes bookmarks if a banned DOMAIN matches (host == domain or endswith ".domain")
  • Deletes bookmarks if a banned KEYWORD appears in the DOMAIN, URL SLUG (path), or TITLE
    (exception: NEVER delete *.wordpress.com)
  • Deletes all YouTube links (youtube.com) and short YouTube links (youtu.be, youtube-nocookie.com)
  • Removes duplicate bookmarks (by normalized URL)
  • Merges sibling folders like "Imported", "Imported (1)", "Imported (2)"
  • Categorizes the remaining bookmarks into folders/subfolders (domain routing first, then keywords)
  • Adds Firefox-compatible TAGS to each <A> (category, subcategory, host, etc.). Chrome ignores TAGS safely.
  • Prunes empty folders
  • Prints the top 20 domains (remaining) with counts
  • Outputs a Netscape-format HTML that both Chrome and Firefox can import
"""

from __future__ import annotations

import collections
import html
import os
import re
import sys
import time
from html.parser import HTMLParser
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, unquote


# ----------------------------
# Configuration: Delete rules
# ----------------------------
# Domains to delete (host == domain or host endswith ".domain")
DELETE_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "news.ycombinator.com",
    "wordpress.org",
    # YouTube and common YouTube short/embedded domains
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
}

# Keywords/phrases to delete if found specifically in:
#   - title (case-insensitive)
#   - host (domain)
#   - path (URL "slug")
# We intentionally do NOT check the query string for keyword matches.
DELETE_KEYWORDS = [
    "xdaforum", "xdaforums", "xda-developers",
    "jekyll", "jekyll-theme",
    "wordpress-theme", "wordpress plugin", "wordpress-plugin",
    "how-to", "how to",
    "static site", "static site generator",
    "flat file", "flat file cms",
    "getkirby", "kirby cms", "kirby theme", "kirby plugin",
    "statamic theme", "statamic plugin", "statamic",
    "torrent",
    "python",
    "foobar",
    "codejam",
    "hackerrank",
]

# Exception: never delete anything hosted on wordpress.com or its subdomains
def is_wordpress_com_host(host: str) -> bool:
    host = (host or "").lower()
    return host == "wordpress.com" or host.endswith(".wordpress.com")


# ----------------------------
# Configuration: Domain routing (categories that override keyword rules)
# ----------------------------
SPECIAL_DOMAIN_CATEGORIES = {
    "archive.org": "WebArchives",
    "github.com": "github",
    "news.ycombinator.com": "HN",
    "ycombinator.com": "HN",
    "goodreads.com": "goodreads",
}

# Social media → each gets its own folder by name (when not deleted by rules above)
SOCIAL_DOMAINS = {
    "facebook.com": "Facebook",
    "m.facebook.com": "Facebook",
    "twitter.com": "Twitter",
    "x.com": "Twitter",
    "pinterest.com": "Pinterest",
    "reddit.com": "Reddit",
    "instagram.com": "Instagram",
    "linkedin.com": "LinkedIn",
    "tiktok.com": "TikTok",
    "snapchat.com": "Snapchat",
    "threads.net": "Threads",
    "discord.com": "Discord",
    "discord.gg": "Discord",
    "mastodon.social": "Mastodon",
    "medium.com": "Medium",
    "dev.to": "dev.to",
    "hashnode.com": "Hashnode",
}


# ----------------------------
# Configuration: Keyword category rules
# ----------------------------
# Matching is done on (title + URL) strings; for deletion we have stricter host/path/title checks.
CATEGORY_RULES = {
    "Programming": {
        "keywords": [
            "programming", "developer", "coding", "regex", "algorithm",
            "package manager", "sdk", "cli", "lint", "formatter",
            "stackoverflow.com", "stack overflow",
        ],
        "sub": {
            "Python": [
                "python", "pypi.org", "pip", "django", "flask", "fastapi",
                "pandas", "numpy", "scipy", "pytest",
            ],
            "JavaScript": [
                "javascript", " js ", "nodejs.org", "npmjs.com", "webpack", "babel",
                "eslint", "prettier", "jest", "vite",
            ],
            "TypeScript": ["typescript", " ts ", "tsconfig"],
            "React": ["reactjs", "react.dev", "nextjs", "next.js", "create-react-app"],
            "Vue": ["vuejs", "vuejs.org", "nuxt", "nuxtjs"],
            "Svelte": ["svelte.dev", "sveltekit"],
            "PHP": ["php", "laravel", "symfony", "wp-", "wordpress."],
            "Ruby": ["ruby", "rubygems", "rails", "jekyll"],
            "Java": ["java", "spring.io", "maven", "gradle"],
            "C/C++": ["c++", "cppreference", "cplusplus", "gcc", "clang", "llvm"],
            "C#/.NET": ["c#", ".net", "dotnet", "nuget.org"],
            "Go": ["golang", "go.dev", "pkg.go.dev"],
            "Rust": ["rust-lang.org", "crates.io", "cargo"],
            "Databases": [
                "sql", "postgres", "postgresql", "mysql", "mariadb", "sqlite",
                "mongodb", "redis", "elasticsearch", "clickhouse", "neo4j",
            ],
            "DevOps": [
                "docker", "kubernetes", "k8s", "terraform", "helm", "ansible",
                "prometheus", "grafana", "jenkins", "github actions", "gitlab ci",
            ],
            "Cloud": [
                "aws.amazon.com", "amazonaws.com", "cloud.google.com", "gcp",
                "azure.microsoft.com", "vercel", "netlify", "cloudflare",
            ],
            "AI/ML": [
                "machine learning", "deep learning", "pytorch", "tensorflow",
                "scikit-learn", "sklearn", "kaggle", "huggingface.co",
            ],
            "Static Sites": ["jekyll", "hugo", "eleventy", "11ty"],
            "CMS": ["wordpress", "drupal", "ghost.org"],
            "Version Control": ["git", "gitlab.com", "bitbucket.org"],
            "Docs/Reference": ["mdn", "rfc", "spec", "readthedocs", "docs."],
        },
    },

    # Puzzles / Math / Games
    "Puzzles": {
        "keywords": ["puzzle", "puzzles", "puzzling"],
        "sub": {
            "Programming Puzzles": [
                "programming puzzle", "code golf", "codegolf", "advent of code",
                "puzzling.stackexchange.com", "foobar with google", "project euler",
                "microsoft programming puzzle", "facebook hiring puzzle",
                "hackerrank", "codewars", "leetcode", "kattis",
            ],
            "Crosswords": ["crossword", "nyt crossword", "guardian crossword"],
            "Logic Puzzles": ["logic puzzle", "nonogram", "picross", "kakuro", "slitherlink"],
        },
    },
    "Math": {
        "keywords": ["math", "mathematics", "algebra", "geometry", "number theory"],
        "sub": {
            "Mathematical Puzzles": [
                "math puzzle", "mathematical puzzle", "project euler", "brilliant.org",
            ],
            "Mathematical Problems": [
                "problem set", "olympiad", "aops", "artofproblemsolving", "imo", "putnam",
            ],
        },
    },
    "Games": {
        "keywords": ["game", "gaming", "board game", "video game"],
        "sub": {
            "Chess": ["chess.com", "lichess.org", "chess"],
            "Puzzle Games": ["puzzle game", "sokoban", "baba is you"],
            "Video Games": ["store.steampowered.com", "steamcommunity.com", "itch.io", "epic games"],
        },
    },

    # Books & related media
    "Books": {
        "keywords": [
            "book ", " books", "/book/", "/books/",
            "reading list", "to-read", "bookclub", "book club",
            "isbn", "publisher", "ebook", "audiobook", "bookstore",
        ]
    },
    "Book Review": {
        "keywords": [
            "book review", "review of the book", "literary review", "book critique",
            "goodreads review", "nyt book review", "kirkus reviews",
        ]
    },
    "Comics": {
        "keywords": [
            "comic", "comics", "webcomic", "webtoons", "graphic novel",
            "marvel", "dc comics", "image comics", "dark horse",
        ]
    },
    "Manga": {
        "keywords": ["manga", "mangadex", "mangaplus", "mangaupdates", "shonen jump", "viz.com"]
    },
    "Anime": {
        "keywords": [
            "anime", "anilist.co", "myanimelist", "myanimelist.net", "crunchyroll",
            "funimation", "hidive", "aniwave", "anidb",
        ]
    },
    "Graphic Novels": {
        "keywords": ["graphic novel", "graphic novels", "gn review", "trade paperback"]
    },

    "Blogs": {
        "keywords": ["blog", "medium.com", "dev.to", "hashnode", "substack", "wordpress.com"]
    },
    "News": {
        "keywords": [
            "news", "nytimes.com", "bbc.", "cnn.com", "theguardian.com",
            "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "wsj.com"
        ]
    },
    "Travel": {
        "keywords": [
            "travel", "trip", "flight", "hotel", "booking", "airbnb", "expedia",
            "skyscanner", "tripadvisor", "lonelyplanet"
        ],
        "sub": {
            "Thailand": ["thailand", "bangkok", "chiang mai", "phuket"],
            "Vietnam": ["vietnam", "hanoi", "ho chi minh", "saigon", "da nang", "danang"],
            "Japan": ["japan", "tokyo", "kyoto", "osaka"],
            "USA": ["usa", "new york", "san francisco", "los angeles", "california"],
            "UK": ["uk", "london", "england", "britain"],
        },
    },
    "Shopping": {"keywords": ["amazon.", "ebay.", "aliexpress", "etsy", "shop", "store"]},
    "Video": {"keywords": ["youtube.com", "vimeo.com", "twitch.tv", "tiktok.com"]},
    "Music": {"keywords": ["spotify.com", "soundcloud.com", "bandcamp.com"]},
    "Education": {"keywords": ["duolingo", "coursera", "edx.org", "khanacademy", "udemy", "brilliant.org"]},
    "Docs/Reference": {"keywords": ["docs.", "readthedocs", "wikipedia.org", "mdn", "spec", "rfc"]},
    "Design": {"keywords": ["dribbble", "behance", "figma", "font", "palette", "color tool"]},
    "Finance": {"keywords": ["bank", "finance", "investment", "trading", "coinbase", "robinhood", "binance", "stripe", "paypal"]},
    "Productivity": {"keywords": ["notion.so", "trello.", "asana.", "todoist", "calendar", "slack.com", "zoom.us", "meet.google.com"]},
    "Tools": {"keywords": ["converter", "generator", "tool", "utility"]},
}


# ----------------------------
# Data structures
# ----------------------------
class BookmarkItem:
    def __init__(self, title: str, href: str, attrs: Dict[str, str]):
        self.title = title or ""
        self.href = href or ""
        self.attrs = dict(attrs or {})

    def __repr__(self):
        return f"BookmarkItem(title={self.title!r}, href={self.href!r})"


class BookmarkFolder:
    def __init__(self, name: str, attrs: Optional[Dict[str, str]] = None):
        self.name = name or "Untitled"
        self.attrs = dict(attrs or {})
        self.items: List[BookmarkItem] = []
        self.folders: List["BookmarkFolder"] = []

    def add_folder(self, folder: "BookmarkFolder"):
        self.folders.append(folder)

    def add_item(self, item: BookmarkItem):
        self.items.append(item)

    def __repr__(self):
        return f"BookmarkFolder(name={self.name!r}, items={len(self.items)}, subfolders={len(self.folders)})"


# ----------------------------
# Parser (Netscape Bookmark format)
# ----------------------------
class NetscapeBookmarkParser(HTMLParser):
    """
    Minimal parser for Bookmark HTML export. Builds a tree of BookmarkFolder/BookmarkItem.
    Handles <DL>/<DT>/<H3> for folders and <DT>/<A> for items.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = BookmarkFolder("Bookmarks")
        self.stack: List[BookmarkFolder] = []
        self.pending_folder: Optional[BookmarkFolder] = None
        self.current_link_attrs: Optional[Dict[str, str]] = None
        self.current_link_text_chunks: List[str] = []
        self.current_folder_attrs: Optional[Dict[str, str]] = None
        self.current_folder_text_chunks: List[str] = []
        self.in_a = False
        self.in_h3 = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        attrs = {k.upper(): (v or "") for k, v in attrs}
        tag_lower = tag.lower()

        if tag_lower == "dl":
            if not self.stack:
                self.stack.append(self.root)
            elif self.pending_folder is not None:
                self.stack.append(self.pending_folder)
                self.pending_folder = None
            else:
                # Extra DLs sometimes appear; keep hierarchy stable
                if self.stack:
                    self.stack.append(self.stack[-1])
                else:
                    self.stack.append(self.root)

        elif tag_lower == "h3":
            self.in_h3 = True
            self.current_folder_attrs = attrs
            self.current_folder_text_chunks = []

        elif tag_lower == "a":
            self.in_a = True
            self.current_link_attrs = attrs
            self.current_link_text_chunks = []

    def handle_endtag(self, tag: str):
        tag_lower = tag.lower()

        if tag_lower == "h3":
            name = "".join(self.current_folder_text_chunks).strip()
            folder = BookmarkFolder(name=name, attrs=self.current_folder_attrs or {})
            parent = self.stack[-1] if self.stack else self.root
            parent.add_folder(folder)
            self.pending_folder = folder
            self.in_h3 = False
            self.current_folder_attrs = None
            self.current_folder_text_chunks = []

        elif tag_lower == "a":
            title = "".join(self.current_link_text_chunks).strip()
            href = (self.current_link_attrs or {}).get("HREF", "")
            item = BookmarkItem(title=title, href=href, attrs=self.current_link_attrs or {})
            parent = self.stack[-1] if self.stack else self.root
            parent.add_item(item)
            self.in_a = False
            self.current_link_attrs = None
            self.current_link_text_chunks = []

        elif tag_lower == "dl":
            if self.stack:
                self.stack.pop()

    def handle_data(self, data: str):
        if self.in_h3:
            self.current_folder_text_chunks.append(data)
        elif self.in_a:
            self.current_link_text_chunks.append(data)


# ----------------------------
# Utilities
# ----------------------------
def now_epoch() -> int:
    return int(time.time())


def normalize_url(url: str) -> str:
    """Normalize URL for duplicate detection."""
    if not url:
        return ""
    try:
        p = urlparse(url.strip())
        scheme = p.scheme.lower()
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        # strip default ports
        if ":" in netloc:
            host, _, port = netloc.partition(":")
            if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
                netloc = host
            netloc = netloc
        path = unquote(p.path or "/")
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        # ignore fragment; keep query (different queries considered different)
        return f"{netloc}{path}?{p.query}" if p.query else f"{netloc}{path}"
    except Exception:
        return url.strip().lower()


def get_netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def get_path(url: str) -> str:
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def walk_folders(folder: BookmarkFolder):
    yield folder
    for f in folder.folders:
        yield from walk_folders(f)


def flatten_items(folder: BookmarkFolder) -> List[BookmarkItem]:
    out: List[BookmarkItem] = []
    for f in walk_folders(folder):
        out.extend(f.items)
    return out


def dedupe_bookmarks(folder: BookmarkFolder) -> int:
    """Delete duplicates across the whole tree; keep first occurrence."""
    seen = set()
    removed = 0

    def _dedupe(f: BookmarkFolder):
        nonlocal removed
        new_items = []
        for it in f.items:
            key = normalize_url(it.href)
            if key and key in seen:
                removed += 1
                continue
            if key:
                seen.add(key)
            new_items.append(it)
        f.items = new_items
        for sf in f.folders:
            _dedupe(sf)

    _dedupe(folder)
    return removed


def _base_folder_name(name: str) -> str:
    """Normalize folder name for 'Imported (1)' style merging."""
    base = re.sub(r"\s*\(\d+\)\s*$", "", name.strip(), flags=re.IGNORECASE)
    return base.lower()


def merge_similar_sibling_folders(folder: BookmarkFolder) -> int:
    """
    Merge siblings whose names only differ by a ' (n)' suffix (case-insensitive).
    Returns number of merges performed under this folder and descendants.
    """
    merges = 0

    groups: Dict[str, BookmarkFolder] = {}
    new_subs: List[BookmarkFolder] = []
    for sf in folder.folders:
        key = _base_folder_name(sf.name)
        if key not in groups:
            groups[key] = sf
            new_subs.append(sf)
        else:
            target = groups[key]
            target.items.extend(sf.items)
            target.folders.extend(sf.folders)
            merges += 1
    folder.folders = new_subs

    for sf in folder.folders:
        merges += merge_similar_sibling_folders(sf)

    return merges


def category_hit(haystack: str, needles: List[str]) -> bool:
    h = f" {haystack.lower()} "
    for n in needles:
        n_l = n.lower()
        if n_l in h:
            return True
    return False


def domain_category(netloc: str) -> Optional[str]:
    """Return category name for known domains, or None."""
    host = (netloc or "").lower()
    for dom, cat in SPECIAL_DOMAIN_CATEGORIES.items():
        if host == dom or host.endswith("." + dom):
            return cat
    for dom, cat in SOCIAL_DOMAINS.items():
        if host == dom or host.endswith("." + dom):
            return cat
    return None


def should_delete(item: BookmarkItem) -> bool:
    """
    True if the bookmark should be deleted per DELETE_DOMAINS / DELETE_KEYWORDS,
    honoring the wordpress.com exception and YouTube rules.
    """
    url = item.href or ""
    title = (item.title or "").lower()
    host = get_netloc(url)
    host_no_www = host[4:] if host.startswith("www.") else host
    path = (get_path(url) or "").lower()

    # Exception: never delete wordpress.com / *.wordpress.com
    if is_wordpress_com_host(host):
        return False

    # Domain-based delete
    for dom in DELETE_DOMAINS:
        if host_no_www == dom or host_no_www.endswith("." + dom):
            return True

    # Keyword-based delete: check title, host (domain), and path (slug) only
    host_l = host_no_www.lower()
    for kw in DELETE_KEYWORDS:
        kw_l = kw.lower()
        if (kw_l in title) or (kw_l in host_l) or (kw_l in path):
            return True

    return False


# ----------------------------
# Categorization & tagging
# ----------------------------
def categorize_item(item: BookmarkItem) -> Tuple[str, Optional[str]]:
    """
    Return (category, subcategory or None).
    Priority:
      1) Domain-based (archive.org/WebArchives, github, HN, Socials, goodreads)
      2) Keyword-based subcategories
      3) Keyword-based categories
      4) Heuristic for Blogs
      5) Uncategorized
    """
    text = f"{item.title} {item.href}".lower()
    netloc = get_netloc(item.href)

    # 1) Domain-based routing
    dom_cat = domain_category(netloc)
    if dom_cat:
        return dom_cat, None

    # 2) Subcategory matches
    for cat, spec in CATEGORY_RULES.items():
        for sub, kws in spec.get("sub", {}).items():
            if category_hit(text, kws):
                return cat, sub

    # 3) Category-level matches
    for cat, spec in CATEGORY_RULES.items():
        if category_hit(text, spec.get("keywords", [])):
            return cat, None

    # 4) Heuristic for blogs
    if "://blog." in item.href.lower() or "/blog/" in item.href.lower():
        return "Blogs", None

    # 5) Default
    return "Uncategorized", None


def compute_tags(item: BookmarkItem, category: str, subcategory: Optional[str]) -> str:
    """
    Build a Firefox-compatible TAGS string for the <A> element.
    Include: category, subcategory (if any), and the host (without www).
    """
    tags: List[str] = []
    if category and category != "Uncategorized":
        tags.append(category)
    if subcategory:
        tags.append(subcategory)
    host = get_netloc(item.href)
    if host.startswith("www."):
        host = host[4:]
    if host:
        tags.append(host)
    # Deduplicate preserve order
    tags = list(dict.fromkeys([t.strip() for t in tags if t.strip()]))
    return ", ".join(tags)


def build_categorized_tree(items: List[BookmarkItem]) -> BookmarkFolder:
    root = BookmarkFolder("Bookmarks", attrs={"ADD_DATE": str(now_epoch())})
    cat_map: Dict[str, BookmarkFolder] = {}
    sub_map: Dict[Tuple[str, str], BookmarkFolder] = {}

    def ensure_cat(name: str) -> BookmarkFolder:
        if name not in cat_map:
            cat_folder = BookmarkFolder(name, attrs={"ADD_DATE": str(now_epoch())})
            cat_map[name] = cat_folder
            root.add_folder(cat_folder)
        return cat_map[name]

    def ensure_sub(cat_name: str, sub_name: str) -> BookmarkFolder:
        key = (cat_name, sub_name)
        if key not in sub_map:
            cat_folder = ensure_cat(cat_name)
            sub_folder = BookmarkFolder(sub_name, attrs={"ADD_DATE": str(now_epoch())})
            sub_map[key] = sub_folder
            cat_folder.add_folder(sub_folder)
        return sub_map[key]

    for it in items:
        cat, sub = categorize_item(it)
        # Attach Firefox TAGS here so writer can include them
        it.attrs["TAGS"] = compute_tags(it, cat, sub)
        if sub:
            ensure_sub(cat, sub).add_item(it)
        else:
            ensure_cat(cat).add_item(it)

    # Prune empty category folders
    root.folders = [cf for cf in root.folders if (cf.items or cf.folders)]
    return root


def prune_empty_folders(folder: BookmarkFolder) -> None:
    """Recursively delete empty directories."""
    new_subs: List[BookmarkFolder] = []
    for sf in folder.folders:
        prune_empty_folders(sf)
        if sf.items or sf.folders:
            new_subs.append(sf)
    folder.folders = new_subs


# ----------------------------
# Writer (Netscape format)
# ----------------------------
def _attrs_to_html(attrs: Dict[str, str], first=None) -> str:
    ordered = []
    if first and first.upper() in attrs:
        ordered.append((first.upper(), attrs[first.upper()]))
    for k, v in attrs.items():
        if first and k.upper() == first.upper():
            continue
        ordered.append((k, v))
    parts = []
    for k, v in ordered:
        if v is None:
            continue
        parts.append(f'{k}="{html.escape(str(v), quote=True)}"')
    return " ".join(parts)


def write_netscape_html(root: BookmarkFolder, out_path: str):
    lines: List[str] = []
    push = lines.append

    push('<!DOCTYPE NETSCAPE-Bookmark-file-1>')
    push('<!-- This is an automatically generated file.')
    push('     It will be read and overwritten.')
    push('     DO NOT EDIT! -->')
    push('<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">')
    push('<TITLE>Bookmarks</TITLE>')
    push('<H1>Bookmarks</H1>')

    def emit_folder_content(f: BookmarkFolder, level: int):
        indent = "  " * level
        h3_attrs = dict(f.attrs)
        if "ADD_DATE" not in h3_attrs:
            h3_attrs["ADD_DATE"] = str(now_epoch())
        push(f'{indent}<DT><H3 {_attrs_to_html(h3_attrs)}>{html.escape(f.name)}</H3>')
        push(f'{indent}<DL><p>')
        for it in f.items:
            a_attrs = dict(it.attrs)
            a_attrs["HREF"] = it.href
            if "ADD_DATE" not in a_attrs:
                a_attrs["ADD_DATE"] = str(now_epoch())
            # TAGS are already on it.attrs["TAGS"] (Firefox will read them; Chrome ignores)
            push(f'{indent}  <DT><A {_attrs_to_html(a_attrs, first="HREF")}>{html.escape(it.title)}</A>')
        for sf in f.folders:
            emit_folder_content(sf, level + 1)
        push(f'{indent}</DL><p>')

    push('<DL><p>')
    for it in root.items:
        a_attrs = dict(it.attrs)
        a_attrs["HREF"] = it.href
        if "ADD_DATE" not in a_attrs:
            a_attrs["ADD_DATE"] = str(now_epoch())
        push(f'  <DT><A {_attrs_to_html(a_attrs, first="HREF")}>{html.escape(it.title)}</A>')

    for sf in root.folders:
        emit_folder_content(sf, 1)
    push('</DL><p>')

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ----------------------------
# Filtering helpers
# ----------------------------
def filter_delete_bookmarks(folder: BookmarkFolder) -> int:
    """Remove items matching should_delete(). Returns count removed."""
    removed = 0

    def _filter(f: BookmarkFolder):
        nonlocal removed
        kept = []
        for it in f.items:
            if should_delete(it):
                removed += 1
            else:
                kept.append(it)
        f.items = kept
        for sf in f.folders:
            _filter(sf)

    _filter(folder)
    return removed


# ----------------------------
# Orchestration (no CLI args)
# ----------------------------
def parse_bookmarks_html(path: str) -> BookmarkFolder:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()
    parser = NetscapeBookmarkParser()
    parser.feed(data)
    return parser.root


def top_domains_report(items: List[BookmarkItem], top_n: int = 20) -> List[Tuple[str, int]]:
    counts = collections.Counter()
    for it in items:
        host = get_netloc(it.href)
        if host.startswith("www."):
            host = host[4:]
        if host:
            counts[host] += 1
    return counts.most_common(top_n)


def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    in_path = os.path.join(script_dir, "bookmarks.html")
    out_path = os.path.join(script_dir, "bookmarks_sorted.html")

    if not os.path.isfile(in_path):
        print("Error: bookmarks.html not found in script directory.", file=sys.stderr)
        sys.exit(1)

    # 1) Parse original
    root = parse_bookmarks_html(in_path)

    # 2) Delete per rules (wordpress.com exception, YouTube, etc.)
    deleted = filter_delete_bookmarks(root)

    # 3) Remove duplicates
    removed_dupes = dedupe_bookmarks(root)

    # 4) Merge similarly named folders
    merges = merge_similar_sibling_folders(root)

    # 5) Gather remaining items to categorize
    remaining_items = flatten_items(root)

    # 6) Print Top 20 domains (AFTER deletions & dedupe)
    print("\nTop 20 domains by remaining bookmarks:")
    for host, cnt in top_domains_report(remaining_items, top_n=20):
        print(f"{host:40s}  {cnt}")

    # 7) Build categorized tree + attach Firefox TAGS
    categorized_root = build_categorized_tree(remaining_items)

    # 8) Prune empty directories
    prune_empty_folders(categorized_root)

    # 9) Write Netscape HTML (Chrome/Firefox importable)
    write_netscape_html(categorized_root, out_path)

    # Summary
    print("\nSummary:")
    print(f"Input:                     {in_path}")
    print(f"Output:                    {out_path}")
    print(f"Deleted by filters:        {deleted}")
    print(f"Removed duplicates:        {removed_dupes}")
    print(f"Merged similar folders:    {merges}")
    print(f"Total bookmarks output:    {len(remaining_items)}")


if __name__ == "__main__":
    main()
