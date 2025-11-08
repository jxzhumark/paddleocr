import os, re, json, base64, mimetypes, tempfile, requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Tuple
from pathlib import Path, PurePosixPath
from pypdf import PdfReader, PdfWriter

@dataclass
class MDArtifact:
    md_text: str
    attachments: Dict[str, bytes]
    meta: Dict[str, Any]

def _is_pdf(path: str) -> bool:
    return str(path).lower().endswith(".pdf")

def _filetype(path: str) -> int:
    return 0 if _is_pdf(path) else 1

def _get_env_url(name: str, default: str) -> str:
    val = (os.getenv(name) or "").strip()
    return default if (not val or val.startswith("${")) else val

def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _norm_relpath(p: str) -> str:
    p = (p or "").strip().lstrip("./")
    parts = [seg for seg in PurePosixPath(p).parts if seg not in ("..", "")]
    return str(PurePosixPath(*parts))

def _pdf_num_pages(pdf_path: str) -> int:
    try:
        return len(PdfReader(pdf_path).pages)
    except Exception:
        return -1

def _split_pdf_range(pdf_path: str, start: int, end: int) -> str:
    reader, writer = PdfReader(pdf_path), PdfWriter()
    n = len(reader.pages)
    for i in range(start, min(end, n)):
        writer.add_page(reader.pages[i])
    out = tempfile.mktemp(suffix=".pdf")
    with open(out, "wb") as f:
        writer.write(f)
    return out

def _chunk_indices(n_pages: int, chunk: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n_pages:
        j = min(i + chunk, n_pages)
        yield i, j
        i = j

def _call_layout_parsing(path: str, api_url: str, token: Optional[str],
                         params: Optional[Dict[str, Any]], file_type: Optional[int]) -> Dict[str, Any]:
    with open(path, "rb") as f:
        file_b64 = base64.b64encode(f.read()).decode("ascii")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"token {token}"
    payload = {"file": file_b64}
    if file_type is not None:
        payload["fileType"] = file_type
    payload.setdefault("visualize", os.getenv("PADDLE_VISUALIZE", "1") != "0")
    payload.setdefault("prettifyMarkdown", True)
    if params:
        payload.update(params)
    timeout = float(os.getenv("PADDLE_TIMEOUT", "120"))
    r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["result"]

def _call_layout_parsing_with_chunking(path: str, api_url: str, token: Optional[str],
                                       params: Optional[Dict[str, Any]], file_type: Optional[int]) -> Dict[str, Any]:
    if not _is_pdf(path):
        return _call_layout_parsing(path, api_url, token, params, file_type)
    n_pages = _pdf_num_pages(path)
    chunk = max(1, int(os.getenv("PADDLE_PDF_CHUNK_PAGES", "16")))
    if n_pages <= 0 or n_pages <= chunk:
        return _call_layout_parsing(path, api_url, token, params, file_type=0)
    merged = []
    for lo, hi in _chunk_indices(n_pages, chunk):
        sub = _split_pdf_range(path, lo, hi)
        try:
            part = _call_layout_parsing(sub, api_url, token, params, file_type=0)
            merged.extend(part.get("layoutParsingResults") or [])
        finally:
            try: os.remove(sub)
            except Exception: pass
    return {"layoutParsingResults": merged}

def _to_page_artifacts(res: Dict[str, Any], assets_prefix_base: str) -> List[MDArtifact]:
    pages = res.get("layoutParsingResults") or []
    arts: List[MDArtifact] = []
    for idx, page in enumerate(pages, start=1):
        md = (page.get("markdown") or {}).get("text") or ""
        images = (page.get("markdown") or {}).get("images") or (page.get("markdown_images") or {})
        attachments: Dict[str, bytes] = {}
        url_map: Dict[str, str] = {}
        fixed_map: Dict[str, str] = {}
        assets_prefix = f"{assets_prefix_base}_p{idx:03d}_assets/"
        for orig_rel, val in (images or {}).items():
            if not isinstance(orig_rel, str) or not orig_rel.strip():
                continue
            rel = _norm_relpath(orig_rel)
            if _is_url(val):
                url_map[rel] = val
                continue
            data_bytes, ext = None, None
            if isinstance(val, str) and val.startswith("data:"):
                try:
                    header, b64 = val.split(",", 1)
                    import mimetypes, base64 as b64m
                    mime = header.split(";", 1)[0].split(":", 1)[1]
                    ext = mimetypes.guess_extension(mime) or ".png"
                    data_bytes = b64m.b64decode(b64)
                except Exception:
                    data_bytes = None
            if data_bytes is None:
                try:
                    import base64 as b64m
                    data_bytes = b64m.b64decode(val)
                    ext = ext or ".png"
                except Exception:
                    continue
            p = PurePosixPath(rel)
            final_suffix = ext or (p.suffix or ".png")
            cand = str(p.with_suffix(final_suffix))
            k = 1
            while cand in attachments:
                cand = str(p.with_name(f"{p.stem}-{k}{final_suffix}"))
                k += 1
            fixed_map[rel] = cand
            attachments[cand] = data_bytes

        def _re_md(m):
            alt, link = m.group(1), m.group(2)
            if link.startswith(("http://", "https://", "data:")):
                return m.group(0)
            key = _norm_relpath(link)
            if key in url_map:
                return f"![{alt}]({url_map[key]})"
            if key in fixed_map:
                return f"![{alt}]({assets_prefix}{fixed_map[key]})"
            return m.group(0)

        def _re_img(m):
            before, src, after = m.group(1), m.group(2), m.group(3)
            if src.startswith(("http://", "https://", "data:")):
                return m.group(0)
            key = _norm_relpath(src)
            if key in url_map:
                return f'<img{before}src="{url_map[key]}"{after}>'
            if key in fixed_map:
                return f'<img{before}src="{assets_prefix}{fixed_map[key]}"{after}>'
            return m.group(0)

        md2 = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _re_md, md)
        md2 = re.sub(r'<img([^>]*?)src="([^"]+)"([^>]*)>', _re_img, md2, flags=re.IGNORECASE)
        arts.append(MDArtifact(md_text=md2, attachments=attachments, meta={"engine": "paddle_layout", "page_index": idx}))
    return arts

def _call_stack(path: str, api: str, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    api_url = api or _get_env_url("PPSV3_URL", "http://127.0.0.1:8080/layout-parsing")
    token = os.getenv("PADDLE_TOKEN")
    return _call_layout_parsing_with_chunking(path, api_url, token, params, file_type=_filetype(path))

def process_file_pages(path: str, prefer: str = "vl_first", api_url: Optional[str] = None,
                       params: Optional[Dict[str, Any]] = None) -> List[MDArtifact]:
    res = _call_stack(path, api_url or _get_env_url("PADDLE_VL_URL", "http://127.0.0.1:8080/layout-parsing"), params)
    stem = Path(path).stem
    return _to_page_artifacts(res, assets_prefix_base=stem)
