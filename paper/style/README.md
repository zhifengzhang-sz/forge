# paper/style/ — light-theme conversion assets

Mirrors `divine-book/style/` layout but uses an **Atom One Light** palette
suitable for academic-paper PDF output (light-on-dark inverted to dark-on-light).

## Files

| File | Purpose |
|------|---------|
| `atom-one-light.css` | Light-theme CSS for HTML→PDF conversion via weasyprint. |
| `mermaid-theme.txt` | One-liner `%%{init:…}%%` block to paste at the top of any mermaid code block. Light-palette equivalent of divine-book's `mermaid-theme.txt`. |

## Color palette (Atom One Light)

| Role | Hex |
|------|-----|
| Background | `#fafafa` (page) / `#ffffff` (mermaid canvas) |
| Foreground / body text | `#383a42` |
| Headings | `#000000` |
| Links | `#4078f2` (blue) |
| Strong / emphasis | `#c18401` (yellow-orange) |
| Code foreground | `#986801` (orange) |
| Code background | `#e5e5e6` (light gray) |
| Pre/code-block background | `#f0f0f0` (slightly darker gray) |
| Borders | `#d4d4d4` (light gray) |
| Comment / muted | `#a0a1a7` |
| Mermaid node fill | `#eef1f5` (very light blue-gray) |
| Mermaid cluster bg | `#f5f7fa` (almost white blue-gray) |
| Mermaid line/edge | `#4078f2` (blue) |

## Mermaid styling — pasting the theme

Paste `mermaid-theme.txt`'s contents (the single line) as the **first line**
inside each mermaid code block:

````markdown
```mermaid
%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#eef1f5', ...}}}%%
flowchart LR
    A --> B
```
````

Mermaid does not support global theme configuration in static markdown — each
diagram needs its own `%%{init:…}%%` line.

## Generating PDFs

Use the wrapper scripts at `paper/scripts/`:

```bash
paper/scripts/md-to-html.sh paper/paper.md     # → paper/paper.html
paper/scripts/html-to-pdf.sh paper/paper.html  # → paper/paper.pdf
```

The HTML step uses `atom-one-light.css` from this directory by default.
Override with `FFF_STYLE_DIR=…`.
