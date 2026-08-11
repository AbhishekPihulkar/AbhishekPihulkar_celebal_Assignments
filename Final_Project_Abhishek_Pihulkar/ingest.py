"""
ingest.py
---------
Loads car brochure PDFs from data/, splits them into structured chunks
(by known brochure sections where detectable, else fixed-size fallback),
and attaches metadata: brand, model, section, page_number, doc_version.

Brand/model/version are pulled from the filename convention:
    <brand>_<model>_<version>.pdf
e.g.  hyundai_creta_2026.pdf  ->  brand=hyundai, model=creta, version=2026

Run:  python ingest.py
Output: data/chunks.json  (consumed by vector_store.py)
"""

import os
import re
import json
from pypdf import PdfReader

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "chunks.json")

SECTION_KEYWORDS = {
    "engine and performance": ["engine", "performance", "power", "torque", "horsepower"],
    "mileage and fuel efficiency": ["mileage", "fuel efficiency", "kmpl", "fuel economy"],
    "safety": ["safety", "airbag", "abs", "esc", "crash"],
    "dimensions": ["dimensions", "wheelbase", "ground clearance", "boot space", "length", "width"],
    "interior and comfort": ["interior", "comfort", "seating", "upholstery", "legroom"],
    "infotainment and connectivity": ["infotainment", "touchscreen", "connectivity", "bluetooth", "android auto", "apple carplay"],
}

FIXED_CHUNK_SIZE = 800   
FIXED_CHUNK_OVERLAP = 150


def parse_filename(filename: str):
    """Extract brand, model, version from '<brand>_<model>_<version>.pdf'."""
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    brand = parts[0].title() if len(parts) > 0 else "Unknown"
    model = parts[1].title() if len(parts) > 1 else "Unknown"
    version = parts[2] if len(parts) > 2 else "NA"
    return brand, model, version


def detect_section(text: str) -> str:
    """Heuristic: return the section label whose keywords appear most in this text block."""
    text_lower = text.lower()
    best_section, best_score = "general", 0
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(text_lower.count(kw) for kw in keywords)
        if score > best_score:
            best_section, best_score = section, score
    return best_section


def fixed_chunks(text: str, size=FIXED_CHUNK_SIZE, overlap=FIXED_CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def ingest_pdf(path: str):
    filename = os.path.basename(path)
    brand, model, version = parse_filename(filename)

    reader = PdfReader(path)
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
        if not blocks:
            blocks = [text]

        for block in blocks:
            block = block.strip()
            if len(block) < 40:
                continue  
            section = detect_section(block)
            sub_chunks = fixed_chunks(block) if len(block) > FIXED_CHUNK_SIZE else [block]
            for sub in sub_chunks:
                chunks.append({
                    "text": sub,
                    "metadata": {
                        "brand": brand,
                        "model": model,
                        "doc_version": version,
                        "section": section,
                        "page_number": page_num,
                        "source_file": filename,
                    }
                })

    return chunks


def main():
    all_chunks = []
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/. Add files named like 'hyundai_creta_2026.pdf' and re-run.")
        return

    for f in pdf_files:
        path = os.path.join(DATA_DIR, f)
        print(f"Ingesting {f} ...")
        file_chunks = ingest_pdf(path)
        all_chunks.extend(file_chunks)
        print(f"  -> {len(file_chunks)} chunks")

    with open(OUTPUT_FILE, "w") as fh:
        json.dump(all_chunks, fh, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()