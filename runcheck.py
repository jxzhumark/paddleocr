import os, sys, pathlib
from dotenv import load_dotenv
load_dotenv(override=True)

from src.pipeline import process_file_pages, MDArtifact

def save_artifact(art: MDArtifact, base_stem: str, page_idx: int, out_dir="out") -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{base_stem}_p{page_idx:03d}"
    md_path = os.path.join(out_dir, f"{stem}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(art.md_text or "")
    if art.attachments:
        base_assets = os.path.join(out_dir, f"{stem}_assets")
        for rel, data in art.attachments.items():
            abs_path = os.path.join(base_assets, rel)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "wb") as w:
                w.write(data)
    return md_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python runcheck.py <path-to-file>")
        sys.exit(1)
    src = sys.argv[1]
    base = pathlib.Path(src).stem
    parts = process_file_pages(src, prefer="vl_first")
    print(f"[pages] {len(parts)} page(s)")
    for art in parts:
        md_path = save_artifact(art, base_stem=base, page_idx=art.meta["page_index"])
        print(f"[saved] {md_path}")
