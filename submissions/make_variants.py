"""Generate TMLR (anonymous) and IEEE (IEEEtran) variants of paper/main.tex."""
import re
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE.parent / "paper" / "main.tex").read_text(encoding="utf-8")
BEGIN, BIBSTYLE, APPX = r"\begin{document}", r"\bibliographystyle", r"\appendix"
body = src.split(BEGIN)[1].split(BIBSTYLE)[0]
appendix = src.split(APPX, 1)[1]
TITLE = ("DOTS: Exactly Removable Continual Learning for Frozen Language Models "
         "via Hyperdimensional Sparse Memory")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
after_title = body.split(r"\maketitle", 1)[1].lstrip()

COMMON_PKGS = r"""\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{url}
\usepackage{algorithm}
\usepackage{algpseudocode}
\newtheorem{proposition}{Proposition}
\newcommand{\sys}{\textsc{Dots}}
"""

# ---------------- TMLR: double-blind ----------------
b = after_title
b = re.sub(r"\\section\*\{Code and Data Availability\}.*?\n\n",
           "\\\\section*{Reproducibility Statement}\nThe code, notebooks and exact fact lists will be released in a public "
           "repository upon acceptance; an anonymized copy is available to reviewers on request. "
           "All data come from public datasets: TriviaQA, NaturalQuestions-open and WikiText-2.\n\n", b, flags=re.S)
b = re.sub(r"\\section\*\{Use of AI Assistance\}.*?\n\n", "", b, flags=re.S)
b = re.sub(r"\\section\*\{Author Contributions, Conflicts of Interest and Funding\}.*?\n\n", "", b, flags=re.S)
tmlr = (r"\documentclass[10pt]{article}" "\n" r"\usepackage[submit]{tmlr}" "\n" + COMMON_PKGS +
        r"\usepackage{hyperref}" "\n"
        rf"\title{{{TITLE}}}" "\n"
        r"\author{\name Anonymous authors \email Paper under double-blind review}" "\n"
        r"\def\month{MM}\def\year{YYYY}\def\openreview{\url{https://openreview.net/forum?id=XXXX}}" "\n"
        r"\begin{document}" "\n\\maketitle\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n" +
        re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}\n?", "", b, flags=re.S).rstrip() +
        "\n\n\\bibliography{references}\n\\bibliographystyle{tmlr}\n\n\\appendix" + appendix)
(HERE / "tmlr" / "main.tex").write_text(tmlr, encoding="utf-8")

# ---------------- IEEE: IEEEtran journal, two column ----------------
i = after_title
i = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}\n?", "", i, flags=re.S)
i = i.replace("\\begin{table}[t]\n\\centering\n\\small",
              "\\begin{table*}[t]\n\\centering\n\\small").replace("\\end{table}", "\\end{table*}")
i = i.replace("\\begin{minipage}[t]{0.48\\linewidth}", "\\begin{minipage}[t]{0.48\\textwidth}")
i = i.replace("\\begin{figure}[t]", "\\begin{figure*}[t]").replace("\\end{figure}", "\\end{figure*}")
ieee = (r"\documentclass[journal]{IEEEtran}" "\n" + COMMON_PKGS + r"\usepackage{cite}" "\n" r"\let\citep\cite \let\citet\cite" "\n" r"\usepackage[hidelinks]{hyperref}" "\n"
        r"\begin{document}" "\n"
        rf"\title{{{TITLE}}}" "\n"
        r"\author{Sohan Merugu\thanks{S. Merugu is an independent researcher (e-mail: sohanmerugu@gmail.com).}}" "\n"
        r"\markboth{S. Merugu: DOTS}{S. Merugu: DOTS}" "\n\\maketitle\n"
        "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n"
        "\\begin{IEEEkeywords}\nContinual learning, machine unlearning, sparse memory, large language models, "
        "catastrophic forgetting, hyperdimensional computing.\n\\end{IEEEkeywords}\n\n" + i.rstrip() +
        "\n\n\\bibliographystyle{IEEEtran}\n\\bibliography{references}\n\n\\appendix" + appendix)
(HERE / "ieee" / "main.tex").write_text(ieee, encoding="utf-8")
print("written tmlr/main.tex and ieee/main.tex")
