"""Create final submission files for the loan default project."""

from __future__ import annotations

import html
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "Loan_Default_Prediction_System.ipynb"
SOURCE_SUMMARY = ROOT / "loan_default_summary.md"

FINAL_NOTEBOOK = ROOT / "Final_Loan_Default_Prediction_System.ipynb"
FINAL_HTML = ROOT / "Final_Loan_Default_Prediction_System.html"
FINAL_SUMMARY = ROOT / "Final_Loan_Default_Summary.md"


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped


def markdown_to_html(source: str) -> str:
    lines = source.splitlines()
    parts: list[str] = []
    in_list = False
    paragraph: list[str] = []

    def close_paragraph():
        if paragraph:
            parts.append(f"<p>{inline_markdown(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            close_paragraph()
            close_list()
            continue

        if stripped.startswith("#"):
            close_paragraph()
            close_list()
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            title = stripped[level:].strip()
            parts.append(f"<h{level}>{inline_markdown(title)}</h{level}>")
        elif stripped.startswith("- "):
            close_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append(f"<li>{inline_markdown(stripped[2:])}</li>")
        else:
            paragraph.append(stripped)

    close_paragraph()
    close_list()
    return "\n".join(parts)


def output_to_html(output: dict) -> str:
    output_type = output.get("output_type")
    if output_type == "stream":
        return f"<pre class=\"output\">{html.escape(''.join(output.get('text', [])))}</pre>"
    if output_type in {"display_data", "execute_result"}:
        data = output.get("data", {})
        if "text/html" in data:
            return f"<div class=\"output-html\">{data['text/html']}</div>"
        if "text/plain" in data:
            text_plain = data["text/plain"]
            if isinstance(text_plain, list):
                text_plain = "".join(text_plain)
            return f"<pre class=\"output\">{html.escape(str(text_plain))}</pre>"
    if output_type == "error":
        traceback = "\n".join(output.get("traceback", []))
        return f"<pre class=\"output error\">{html.escape(traceback)}</pre>"
    return ""


def notebook_to_html() -> str:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    body_parts = []

    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            body_parts.append(f"<section class=\"markdown-cell\">{markdown_to_html(source)}</section>")
        elif cell["cell_type"] == "code":
            body_parts.append(
                "<section class=\"code-cell\">"
                f"<pre class=\"code\"><code>{html.escape(source)}</code></pre>"
                + "".join(output_to_html(output) for output in cell.get("outputs", []))
                + "</section>"
            )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Loan Default Prediction System Notebook</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      color: #17202a;
      line-height: 1.55;
      margin: 0;
      background: #f6f8fb;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 24px 56px;
      background: #ffffff;
      min-height: 100vh;
    }}
    h1, h2, h3 {{ color: #17324d; }}
    .code-cell, .markdown-cell {{
      margin: 18px 0;
    }}
    pre.code {{
      background: #101820;
      color: #f7fafc;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 13px;
    }}
    pre.output {{
      background: #f1f5f9;
      border-left: 4px solid #2f6f73;
      padding: 12px;
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d7dee8;
      padding: 7px 9px;
      text-align: left;
    }}
    th {{ background: #edf2f7; }}
    svg {{ max-width: 100%; height: auto; }}
    code {{
      background: #eef2f7;
      padding: 1px 4px;
      border-radius: 3px;
    }}
  </style>
</head>
<body>
<main>
{''.join(body_parts)}
</main>
</body>
</html>
"""


def main():
    shutil.copyfile(SOURCE_NOTEBOOK, FINAL_NOTEBOOK)
    shutil.copyfile(SOURCE_SUMMARY, FINAL_SUMMARY)
    FINAL_HTML.write_text(notebook_to_html(), encoding="utf-8")

    print(f"Created {FINAL_NOTEBOOK.name}")
    print(f"Created {FINAL_HTML.name}")
    print(f"Created {FINAL_SUMMARY.name}")


if __name__ == "__main__":
    main()
