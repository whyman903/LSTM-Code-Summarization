from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TRAIN_SIZE = 50_000
DEFAULT_VAL_SIZE = 1_000
DEFAULT_MAX_REPOS = 700
DEFAULT_MIN_STARS = 100
DEFAULT_MAX_JAVA_FILES_PER_REPO = 120
DEFAULT_MAX_PAIRS_PER_REPO = 800
DEFAULT_MIN_CODE_TOKENS = 5
DEFAULT_MAX_CODE_TOKENS = 400
DEFAULT_MIN_SUMMARY_TOKENS = 3
DEFAULT_MAX_SUMMARY_TOKENS = 40
DEFAULT_CLONE_WORKERS = 8
DEFAULT_REPO_BATCH_SIZE = 20
DEFAULT_SEED = 42

CACHE_DIR = Path(".dataset_cache")
CLONE_DIR = CACHE_DIR / "java_repos"
REPO_LIST_FILE = CACHE_DIR / "repo_list.csv"
DEFAULT_OUTPUT_DIR = Path("artifacts")

EXCLUDED_DIR_NAMES = {
    ".git",
    ".github",
    ".idea",
    ".mvn",
    ".svn",
    ".gradle",
    "build",
    "build-tools",
    "coverage",
    "demo",
    "demos",
    "dist",
    "doc",
    "docs",
    "example",
    "examples",
    "generated",
    "node_modules",
    "out",
    "sample",
    "samples",
    "site",
    "target",
    "test",
    "tests",
    "tmp",
    "vendor",
}

JAVA_TOKEN_RE = re.compile(
    r"[A-Za-z_$][\w$]*|\d+|==|!=|<=|>=|&&|\|\||<<|>>|>>>|[{}()\[\].,;:+\-*/%<>!=?&|^~]"
)
JAVADOC_RE = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
METHOD_NAME_RE = re.compile(r"([A-Za-z_$][\w$]*)\s*$")
INLINE_TAG_WITH_TEXT_RE = re.compile(r"\{@\w+\s+([^}]+)\}")
INLINE_TAG_RE = re.compile(r"\{@[^}]+\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
NON_ENGLISH_RE = re.compile(r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
SUMMARY_TAG_PREFIXES = ("@param", "@return", "@throws", "@see", "@since", "@deprecated")
CONTROL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "return",
    "throw",
    "new",
    "synchronized",
}
MODIFIER_PREFIX_RE = re.compile(
    r"^(?:public|protected|private|static|final|abstract|synchronized|native|strictfp|default)\b"
)


def log(message: str) -> None:
    print(message, flush=True)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def github_get_json(url: str, token: Optional[str]) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "lstm-code-summarization-dataset-builder",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code == 403:
            raise RuntimeError(
                "GitHub API rate limit reached. Set GITHUB_TOKEN and rerun.\n"
                f"Response: {detail[:400]}"
            ) from exc
        raise RuntimeError(f"GitHub API request failed ({exc.code}): {detail[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc}") from exc


def load_or_fetch_repos(
    max_repos: int,
    min_stars: int,
    token: Optional[str],
    refresh_repo_list: bool,
) -> List[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if REPO_LIST_FILE.exists() and not refresh_repo_list:
        with REPO_LIST_FILE.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            repos = list(reader)
        if len(repos) >= max_repos:
            log(f"Loaded {len(repos)} cached repositories from {REPO_LIST_FILE}")
            return repos[:max_repos]
        if repos:
            log(
                f"Cached repository index only has {len(repos)} repos; "
                f"refreshing to reach {max_repos}."
            )

    repos: List[dict] = []
    page = 1
    per_page = 100
    query = f"language:java stars:>={min_stars} fork:false archived:false"

    log(f"Fetching up to {max_repos} public Java repositories from GitHub...")
    while len(repos) < max_repos:
        params = urlencode(
            {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "page": page,
                "per_page": per_page,
            }
        )
        url = f"https://api.github.com/search/repositories?{params}"
        data = github_get_json(url, token)
        items = data.get("items", [])
        if not items:
            break

        for item in items:
            if item.get("fork") or item.get("archived"):
                continue
            repos.append(
                {
                    "full_name": item["full_name"],
                    "clone_url": item["clone_url"],
                    "html_url": item["html_url"],
                    "stars": str(item["stargazers_count"]),
                    "description": item.get("description") or "",
                }
            )
            if len(repos) >= max_repos:
                break
        page += 1

    with REPO_LIST_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["full_name", "clone_url", "html_url", "stars", "description"],
        )
        writer.writeheader()
        writer.writerows(repos)
    log(f"Saved repository index to {REPO_LIST_FILE}")
    return repos


def clone_repo(repo: dict) -> Tuple[str, Optional[Path], Optional[str], bool]:
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    repo_name = repo["full_name"]
    dest_dir = CLONE_DIR / repo_name.replace("/", "__")

    if (dest_dir / ".git").exists():
        return repo_name, dest_dir, None, True

    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)

    cmd = ["git", "clone", "--depth", "1", "--quiet", repo["clone_url"], str(dest_dir)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return repo_name, None, "clone timed out", False

    if result.returncode != 0:
        stderr = normalize_whitespace(result.stderr)[:300]
        shutil.rmtree(dest_dir, ignore_errors=True)
        return repo_name, None, stderr or "clone failed", False

    return repo_name, dest_dir, None, False


def clone_repositories(repos: Sequence[dict], workers: int) -> Dict[str, Path]:
    cloned: Dict[str, Path] = {}
    failures: List[Tuple[str, str]] = []

    log(f"Cloning {len(repos)} repositories with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(clone_repo, repo): repo for repo in repos}
        for index, future in enumerate(as_completed(futures), start=1):
            repo = futures[future]
            repo_name, repo_path, error, cached = future.result()
            if repo_path is not None:
                cloned[repo_name] = repo_path
                status = "cached" if cached else "cloned"
                if index % 25 == 0 or index == len(repos):
                    log(f"  {index}/{len(repos)} repos processed ({status}: {repo_name})")
            else:
                failures.append((repo_name, error or "unknown error"))

    log(f"Cloned or reused {len(cloned)} repositories")
    if failures:
        log(f"Failed to clone {len(failures)} repositories")
        for repo_name, error in failures[:10]:
            log(f"  - {repo_name}: {error}")
    return cloned


def should_skip_dir(path: Path) -> bool:
    return any(part.lower() in EXCLUDED_DIR_NAMES for part in path.parts)


def find_java_files(repo_path: Path) -> List[Path]:
    java_files: List[Path] = []
    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        rel_parts = root_path.relative_to(repo_path).parts if root_path != repo_path else ()
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIR_NAMES]
        if any(part.lower() in EXCLUDED_DIR_NAMES for part in rel_parts):
            continue

        for name in files:
            if not name.endswith(".java"):
                continue
            file_path = root_path / name
            if should_skip_dir(file_path.relative_to(repo_path).parent):
                continue
            java_files.append(file_path)
    return java_files


def select_java_files(java_files: Sequence[Path], max_files: int, seed: int, repo_name: str) -> List[Path]:
    files = sorted(java_files, key=lambda p: str(p).lower())
    preferred = [p for p in files if "src/main/java" in str(p).replace("\\", "/").lower()]
    other = [p for p in files if p not in preferred]
    rng = random.Random(f"{seed}:{repo_name}")
    rng.shuffle(preferred)
    rng.shuffle(other)
    selected = preferred[:max_files]
    if len(selected) < max_files:
        selected.extend(other[: max_files - len(selected)])
    return selected


def read_text(path: Path) -> Optional[str]:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def clean_javadoc_summary(raw_comment: str) -> Optional[str]:
    lines: List[str] = []
    saw_summary_text = False
    for raw_line in raw_comment.splitlines():
        line = raw_line.strip()
        if line.startswith("*"):
            line = line[1:].strip()
        if not line:
            if saw_summary_text:
                break
            continue
        if line.lower().startswith(SUMMARY_TAG_PREFIXES):
            break
        lines.append(line)
        saw_summary_text = True

    if not lines:
        return None

    text = " ".join(lines)
    # Preserve the visible payload from inline tags like {@code foo} before
    # stripping any remaining tag wrappers.
    text = INLINE_TAG_WITH_TEXT_RE.sub(r"\1", text)
    text = INLINE_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = normalize_whitespace(text)
    if not text:
        return None
    if URL_RE.search(text):
        return None
    if NON_ENGLISH_RE.search(text):
        return None

    text = normalize_whitespace(text).strip(" -")
    if not text:
        return None

    lowered = text.lower()
    if lowered.startswith(("todo", "fixme", "deprecated", "auto generated")):
        return None
    # Filter out auto-generated / IDE boilerplate summaries
    if any(phrase in lowered for phrase in (
        "intellij", "generated by", "auto-generated", "autogenerated",
        "this method was generated", "noinspection",
    )):
        return None
    return lowered


def skip_non_code(source: str, pos: int) -> int:
    while pos < len(source):
        if source[pos].isspace():
            pos += 1
            continue
        if source.startswith("//", pos):
            newline = source.find("\n", pos)
            pos = len(source) if newline == -1 else newline + 1
            continue
        if source.startswith("/*", pos):
            end = source.find("*/", pos + 2)
            pos = len(source) if end == -1 else end + 2
            continue
        break
    return pos


def consume_annotation(signature: str, start: int) -> int:
    pos = start + 1
    while pos < len(signature) and (signature[pos].isalnum() or signature[pos] in {"_", "$", "."}):
        pos += 1

    while pos < len(signature) and signature[pos].isspace():
        pos += 1

    if pos >= len(signature) or signature[pos] != "(":
        return pos

    depth = 0
    in_string: Optional[str] = None
    escape = False

    while pos < len(signature):
        char = signature[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
        else:
            if char in {'"', "'"}:
                in_string = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return pos + 1
        pos += 1
    return pos


def strip_leading_annotations(signature: str) -> str:
    compact = normalize_whitespace(signature).lstrip()
    while compact.startswith("@"):
        end = consume_annotation(compact, 0)
        compact = compact[end:].lstrip()
    return compact


def strip_modifier_prefixes(text: str) -> str:
    compact = normalize_whitespace(text).lstrip()
    while True:
        match = MODIFIER_PREFIX_RE.match(compact)
        if not match:
            return compact
        compact = compact[match.end() :].lstrip()


def strip_leading_type_parameters(text: str) -> str:
    compact = text.lstrip()
    if not compact.startswith("<"):
        return compact

    depth = 0
    pos = 0
    while pos < len(compact):
        char = compact[pos]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                return compact[pos + 1 :].lstrip()
        pos += 1
    return compact


def find_signature_body_open(source: str, pos: int) -> Optional[int]:
    paren_depth = 0
    in_string: Optional[str] = None
    escape = False

    while pos < len(source):
        char = source[pos]
        next_char = source[pos + 1] if pos + 1 < len(source) else ""

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
            pos += 1
            continue

        if char == "/" and next_char == "/":
            newline = source.find("\n", pos)
            if newline == -1:
                return None
            pos = newline + 1
            continue
        if char == "/" and next_char == "*":
            end = source.find("*/", pos + 2)
            if end == -1:
                return None
            pos = end + 2
            continue
        if char in ('"', "'"):
            in_string = char
            pos += 1
            continue
        if char == "(":
            paren_depth += 1
        elif char == ")" and paren_depth > 0:
            paren_depth -= 1
        elif char == ";" and paren_depth == 0:
            return None
        elif char == "{" and paren_depth == 0:
            return pos
        pos += 1

    return None


def find_matching_brace(source: str, open_pos: int) -> Optional[int]:
    depth = 0
    in_string: Optional[str] = None
    in_line_comment = False
    in_block_comment = False
    escape = False
    pos = open_pos

    while pos < len(source):
        char = source[pos]
        next_char = source[pos + 1] if pos + 1 < len(source) else ""

        if in_line_comment:
            if char == "\n":
                in_line_comment = False
            pos += 1
            continue

        if in_block_comment:
            if char == "*" and next_char == "/":
                in_block_comment = False
                pos += 2
            else:
                pos += 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == in_string:
                in_string = None
            pos += 1
            continue

        if char == "/" and next_char == "/":
            in_line_comment = True
            pos += 2
            continue
        if char == "/" and next_char == "*":
            in_block_comment = True
            pos += 2
            continue
        if char in ('"', "'"):
            in_string = char
            pos += 1
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1

    return None


def looks_like_method_signature(signature: str) -> Tuple[bool, Optional[str]]:
    compact = strip_leading_annotations(signature)
    if "(" not in compact or ")" not in compact:
        return False, None
    if any(keyword in compact for keyword in (" class ", " interface ", " enum ", " record ")):
        return False, None
    prefix = compact.split("(", 1)[0]
    if "=" in prefix:
        return False, None
    match = METHOD_NAME_RE.search(prefix)
    if not match:
        return False, None
    method_name = match.group(1)
    prefix_before_name = prefix[: match.start()]
    prefix_before_name = strip_leading_type_parameters(
        strip_modifier_prefixes(prefix_before_name)
    )
    if not prefix_before_name:
        return False, None
    if method_name in CONTROL_KEYWORDS:
        return False, None
    # Filter out IDE-generated method names (e.g. $$$setupUI$$$)
    if "$$$" in method_name:
        return False, None
    return True, method_name


def tokenize_like_java(code: str) -> List[str]:
    return JAVA_TOKEN_RE.findall(code)


def is_trivial_getter_setter(method_name: str, code_token_count: int) -> bool:
    lowered = method_name.lower()
    if lowered.startswith(("get", "set", "is")) and code_token_count <= 25:
        return True
    return False


def extract_pairs_from_file(
    file_path: Path,
    repo_name: str,
    args: argparse.Namespace,
) -> List[dict]:
    source = read_text(file_path)
    if not source:
        return []

    pairs: List[dict] = []
    for match in JAVADOC_RE.finditer(source):
        summary = clean_javadoc_summary(match.group(1))
        if not summary:
            continue

        method_start = skip_non_code(source, match.end())
        body_open = find_signature_body_open(source, method_start)
        if body_open is None:
            continue
        body_close = find_matching_brace(source, body_open)
        if body_close is None:
            continue

        signature = source[method_start:body_open]
        valid, method_name = looks_like_method_signature(signature)
        if not valid or not method_name:
            continue

        method_source = normalize_whitespace(source[method_start : body_close + 1])
        summary = normalize_whitespace(summary)
        code_tokens = tokenize_like_java(method_source)
        summary_tokens = summary.split()

        if len(code_tokens) < args.min_code_tokens:
            continue
        if len(code_tokens) > args.max_code_tokens:
            continue
        if len(summary_tokens) < args.min_summary_tokens:
            continue
        if len(summary_tokens) > args.max_summary_tokens:
            continue
        if is_trivial_getter_setter(method_name, len(code_tokens)):
            continue

        pairs.append(
            {
                "repo": repo_name,
                "file": str(file_path),
                "method_name": method_name,
                "code": method_source,
                "summary": summary,
                "code_token_count": len(code_tokens),
            }
        )
    return pairs


def collect_repo_pairs(
    repo_name: str,
    repo_path: Path,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    java_files = find_java_files(repo_path)
    selected_files = select_java_files(
        java_files,
        max_files=args.max_java_files_per_repo,
        seed=args.seed,
        repo_name=repo_name,
    )

    repo_pairs: List[dict] = []
    seen_codes = set()
    for file_path in selected_files:
        pairs = extract_pairs_from_file(file_path, repo_name, args)
        for pair in pairs:
            if pair["code"] in seen_codes:
                continue
            seen_codes.add(pair["code"])
            repo_pairs.append(pair)
            if len(repo_pairs) >= args.max_pairs_per_repo:
                break
        if len(repo_pairs) >= args.max_pairs_per_repo:
            break

    rng = random.Random(f"pairs:{args.seed}:{repo_name}")
    rng.shuffle(repo_pairs)
    repo_pairs = repo_pairs[: args.max_pairs_per_repo]

    repo_meta = {
        "repo": repo_name,
        "repo_path": str(repo_path),
        "java_files_found": len(java_files),
        "java_files_selected": len(selected_files),
        "pairs_collected": len(repo_pairs),
    }
    return repo_pairs, repo_meta


def build_repo_samples(
    repo_paths: Dict[str, Path],
    repos_by_name: Dict[str, dict],
    global_seen_codes: set,
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    repo_samples: Dict[str, List[dict]] = {}
    repo_metadata: List[dict] = []

    for index, repo_name in enumerate(sorted(repo_paths), start=1):
        pairs, meta = collect_repo_pairs(repo_name, repo_paths[repo_name], args)
        unique_pairs: List[dict] = []
        removed_duplicates = 0
        for pair in pairs:
            code = pair["code"]
            if code in global_seen_codes:
                removed_duplicates += 1
                continue
            global_seen_codes.add(code)
            unique_pairs.append(pair)

        meta["pairs_after_global_dedupe"] = len(unique_pairs)
        meta["pairs_removed_as_cross_repo_duplicates"] = removed_duplicates
        repo_record = dict(repos_by_name[repo_name])
        repo_record.update(meta)
        repo_metadata.append(repo_record)

        if unique_pairs:
            repo_samples[repo_name] = unique_pairs
        if index % 10 == 0 or index == len(repo_paths):
            log(
                f"Processed {index}/{len(repo_paths)} repos | "
                f"repos with usable pairs so far: {len(repo_samples)}"
            )

    return repo_samples, repo_metadata


def split_repo_disjoint(
    repo_samples: Dict[str, List[dict]],
    train_size: int,
    val_size: int,
    seed: int,
) -> Tuple[List[dict], List[dict], Dict[str, str]]:
    repo_names = list(repo_samples)
    rng = random.Random(seed)
    rng.shuffle(repo_names)

    if not repo_names:
        return [], [], {}

    shuffled_samples: Dict[str, List[dict]] = {}
    for repo_name in repo_names:
        samples = list(repo_samples[repo_name])
        random.Random(f"split:{seed}:{repo_name}").shuffle(samples)
        shuffled_samples[repo_name] = samples

    def gather(selected_repo_names: Sequence[str], target_size: int) -> List[dict]:
        collected: List[dict] = []
        for repo_name in selected_repo_names:
            remaining = target_size - len(collected)
            if remaining <= 0:
                break
            collected.extend(shuffled_samples[repo_name][:remaining])
        return collected

    total_target = max(train_size + val_size, 1)
    estimated_val_repo_count = max(1, round(len(repo_names) * val_size / total_target))
    max_val_repo_count = len(repo_names) if len(repo_names) == 1 else len(repo_names) - 1
    estimated_val_repo_count = min(estimated_val_repo_count, max_val_repo_count)

    best_val_repo_count = estimated_val_repo_count
    for candidate_count in range(estimated_val_repo_count, max_val_repo_count + 1):
        candidate_total = sum(len(shuffled_samples[name]) for name in repo_names[:candidate_count])
        best_val_repo_count = candidate_count
        if candidate_total >= val_size:
            break

    val_repo_names = repo_names[:best_val_repo_count]
    train_repo_names = repo_names[best_val_repo_count:]
    repo_to_split = {name: "val" for name in val_repo_names}
    repo_to_split.update({name: "train" for name in train_repo_names})

    train = gather(train_repo_names, train_size)
    val = gather(val_repo_names, val_size)
    return train, val, repo_to_split


def write_lines(path: Path, lines: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def write_outputs(output_dir: Path, train_pairs: Sequence[dict], val_pairs: Sequence[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lines(output_dir / "train_code.txt", [pair["code"] for pair in train_pairs])
    write_lines(output_dir / "train_summary.txt", [pair["summary"] for pair in train_pairs])
    write_lines(output_dir / "val_code.txt", [pair["code"] for pair in val_pairs])
    write_lines(output_dir / "val_summary.txt", [pair["summary"] for pair in val_pairs])


def write_metadata(
    args: argparse.Namespace,
    output_dir: Path,
    repo_metadata: Sequence[dict],
    repo_to_split: Dict[str, str],
    train_pairs: Sequence[dict],
    val_pairs: Sequence[dict],
) -> None:
    metadata_file = output_dir / "dataset_metadata.json"
    metadata = {
        "description": (
            "GitHub-mined Java method summarization dataset. CodeXGLUE/CodeSearchNet "
            "used as a methodological reference, not as a data source."
        ),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "filters": {
            "min_code_tokens": args.min_code_tokens,
            "max_code_tokens": args.max_code_tokens,
            "min_summary_tokens": args.min_summary_tokens,
            "max_summary_tokens": args.max_summary_tokens,
            "drop_trivial_getter_setter": True,
            "drop_constructors": True,
            "drop_url_or_non_english_summaries": True,
            "summary_source": "Javadoc summary section immediately preceding the method (up to the first blank line or @tag).",
        },
        "splits": {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
        },
        "repos": [],
    }

    for repo in repo_metadata:
        row = dict(repo)
        row["split"] = repo_to_split.get(repo["repo"], "unused")
        metadata["repos"].append(row)

    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--max-repos", type=int, default=DEFAULT_MAX_REPOS)
    parser.add_argument("--min-stars", type=int, default=DEFAULT_MIN_STARS)
    parser.add_argument(
        "--max-java-files-per-repo",
        type=int,
        default=DEFAULT_MAX_JAVA_FILES_PER_REPO,
    )
    parser.add_argument(
        "--max-pairs-per-repo",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_REPO,
    )
    parser.add_argument("--min-code-tokens", type=int, default=DEFAULT_MIN_CODE_TOKENS)
    parser.add_argument("--max-code-tokens", type=int, default=DEFAULT_MAX_CODE_TOKENS)
    parser.add_argument(
        "--min-summary-tokens",
        type=int,
        default=DEFAULT_MIN_SUMMARY_TOKENS,
    )
    parser.add_argument(
        "--max-summary-tokens",
        type=int,
        default=DEFAULT_MAX_SUMMARY_TOKENS,
    )
    parser.add_argument("--clone-workers", type=int, default=DEFAULT_CLONE_WORKERS)
    parser.add_argument("--repo-batch-size", type=int, default=DEFAULT_REPO_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--github-token", type=str, default=os.getenv("GITHUB_TOKEN"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh-repo-list", action="store_true")
    parser.add_argument("--allow-smaller-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    repos = load_or_fetch_repos(
        max_repos=args.max_repos,
        min_stars=args.min_stars,
        token=args.github_token,
        refresh_repo_list=args.refresh_repo_list,
    )
    repos_by_name = {repo["full_name"]: repo for repo in repos}
    repo_samples: Dict[str, List[dict]] = {}
    repo_metadata: List[dict] = []
    global_seen_codes = set()
    total_pairs = 0
    stop_after = int((args.train_size + args.val_size) * 1.30)

    for start in range(0, len(repos), args.repo_batch_size):
        batch = repos[start : start + args.repo_batch_size]
        repo_paths = clone_repositories(batch, workers=args.clone_workers)
        if not repo_paths:
            continue

        batch_samples, batch_metadata = build_repo_samples(
            repo_paths,
            repos_by_name,
            global_seen_codes,
            args,
        )
        repo_samples.update(batch_samples)
        repo_metadata.extend(batch_metadata)
        total_pairs += sum(len(samples) for samples in batch_samples.values())
        log(
            f"Candidate-pair total after repos {start + 1}-{start + len(batch)}: "
            f"{total_pairs}"
        )
        if total_pairs >= stop_after:
            log(f"Reached candidate-pair budget ({total_pairs}); stopping early.")
            break

    if not repo_samples:
        raise RuntimeError("No method-summary pairs were extracted from the cloned repositories.")

    log(
        f"Collected {total_pairs} unique candidate pairs across "
        f"{len(repo_samples)} repositories"
    )

    train_pairs, val_pairs, repo_to_split = split_repo_disjoint(
        repo_samples=repo_samples,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    if len(train_pairs) < args.train_size or len(val_pairs) < args.val_size:
        message = (
            f"Dataset smaller than requested: train={len(train_pairs)}/{args.train_size}, "
            f"val={len(val_pairs)}/{args.val_size}. Increase --max-repos or loosen filters."
        )
        if not args.allow_smaller_dataset:
            raise RuntimeError(message)
        log(f"WARNING: {message}")

    train_pairs = train_pairs[: args.train_size]
    val_pairs = val_pairs[: args.val_size]

    write_outputs(output_dir, train_pairs, val_pairs)
    write_metadata(args, output_dir, repo_metadata, repo_to_split, train_pairs, val_pairs)

    log("Wrote dataset files:")
    log(f"  {output_dir / 'train_code.txt'}     {len(train_pairs)}")
    log(f"  {output_dir / 'train_summary.txt'}  {len(train_pairs)}")
    log(f"  {output_dir / 'val_code.txt'}       {len(val_pairs)}")
    log(f"  {output_dir / 'val_summary.txt'}    {len(val_pairs)}")
    log(f"  {output_dir / 'dataset_metadata.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted by user")
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
