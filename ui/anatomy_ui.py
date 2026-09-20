import base64
import re
import sys
from pathlib import Path

import pymupdf
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import orchestrator


PDF_PATH = (PROJECT_ROOT / "assets" / "anatomy.pdf").resolve()


def get_pdf_page_count(pdf_path: Path) -> int:
    try:
        doc = pymupdf.open(str(pdf_path))
        return int(doc.page_count)
    except Exception:
        return 1


def citation_to_pdf_page(citation_page: int) -> int:
    """Return the stored 1-indexed page number for PDF.js."""
    page_num = int(citation_page)
    if page_num < 1:
        return 1
    return page_num


def normalize_pdf_page(page_num: int) -> int:
    page_count = get_pdf_page_count(PDF_PATH)
    if page_num < 1:
        return 1
    if page_num > page_count:
        return page_count
    return int(page_num)


def extract_citations(answer_text: str):
    """Return (answer_text_without_citations, list[dict]) from the answer string."""
    if "Citations:" not in answer_text:
        return answer_text.strip(), []

    answer, citations_blob = answer_text.split("Citations:", 1)
    citations = []
    for line in citations_blob.strip().splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        item = line[1:].strip()
        match = re.search(r"\(p\.\s*(\d+)\)", item)
        if match:
            book_page = int(match.group(1))
            citations.append({
                "label": item,
                "page": book_page,
                "pdf_page": citation_to_pdf_page(book_page),
            })
    return answer.strip(), citations


def build_pdf_viewer_html(pdf_path: Path, page_num: int = 1) -> str:
    """Return an HTML snippet that uses pdf.js to render a local PDF page in the sidebar."""
    if not pdf_path.exists():
        return "<p>The PDF file was not found.</p>"

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    return f"""
    <style>
      body {{ margin: 0; background: #eef2f8; }}
      #pdf-frame {{ width: 100%; min-height: 720px; overflow: auto; background: #eef2f8; }}
      #pdf-panel {{ width: 100%; min-height: 720px; display: flex; align-items: center; justify-content: center; }}
      canvas {{ max-width: 100%; height: auto; border: 1px solid #cbd5e1; box-shadow: 0 4px 12px rgba(0,0,0,.2); background: white; }}
    </style>
    <div id="pdf-frame">
      <div id="pdf-panel">
        <canvas id="pdf-canvas"></canvas>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
      const pdfData = atob("{pdf_b64}");
      const len = pdfData.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {{ bytes[i] = pdfData.charCodeAt(i); }}

      pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
      pdfjsLib.getDocument({{ data: bytes }}).promise.then(function(pdf) {{
        return pdf.getPage({page_num});
      }}).then(function(page) {{
        const scale = 1.1;
        const viewport = page.getViewport({{ scale: scale }});
        const canvas = document.getElementById('pdf-canvas');
        const context = canvas.getContext('2d');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        page.render({{ canvasContext: context, viewport: viewport }});
      }}).catch(function(err) {{
        const canvas = document.getElementById('pdf-canvas');
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.font = '20px Arial';
        ctx.fillText('PDF preview unavailable', 20, 40);
        console.log(err);
      }});
    </script>
    """


st.set_page_config(page_title="Anatomy RAG", page_icon="📖", layout="wide")

if "pdf_page" not in st.session_state:
    st.session_state.pdf_page = 1

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "citations" not in st.session_state:
    st.session_state.citations = []

if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = ""

st.title("Anatomy RAG")

query = st.text_input(
    "Ask an anatomy question",
    value="what is varicose veins why its caused and how can it be cured",
)

if st.button("Ask"):
    answer_with_citations = orchestrator.run_orchestrator(query, k=3)
    answer, citations = extract_citations(answer_with_citations)
    st.session_state.answer = answer
    st.session_state.citations = citations

with st.sidebar:
    st.subheader("Source PDF")
    st.session_state.pdf_page = normalize_pdf_page(st.session_state.pdf_page)
    st.caption(f"PDF page {st.session_state.pdf_page}")
    if st.session_state.selected_citation:
        st.caption(f"Selected citation: {st.session_state.selected_citation}")
    html = build_pdf_viewer_html(PDF_PATH, page_num=st.session_state.pdf_page)
    components.html(html, height=760, scrolling=True)

if st.session_state.answer:
    st.subheader("Answer")
    st.write(st.session_state.answer)

    st.subheader("Citations")
    for idx, citation in enumerate(st.session_state.citations):
        if st.button(citation["label"], key=f"cit_{idx}"):
            st.session_state.selected_citation = citation["label"]
            st.session_state.pdf_page = int(citation["pdf_page"])
            st.rerun()
else:
    st.info("Ask a question to retrieve the answer and citations.")
