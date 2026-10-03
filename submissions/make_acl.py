"""Generate the ACL Rolling Review (anonymous) variant of paper/main.tex."""
import re
from pathlib import Path

HERE = Path(__file__).parent
src = (HERE.parent / "paper" / "main.tex").read_text(encoding="utf-8")
body = src.split(r"\begin{document}")[1].split(r"\bibliographystyle")[0]
appendix = src.split(r"\appendix", 1)[1]
TITLE = ("DOTS: Exactly Removable Continual Learning for Frozen Language Models "
         "via Hyperdimensional Sparse Memory")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1).strip()
b = body.split(r"\maketitle", 1)[1]
b = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}\n?", "", b, flags=re.S)

# move the algorithm and the ablation table to the appendix (main text is limited to 8 pages)
moved = []
for pat in (r"\\begin\{algorithm\}\[t\].*?\\end\{algorithm\}\n?",
            r"\\begin\{table\}\[t\]\n\\centering\n\\small\n\\caption\{Single-Dot ablations.*?\\end\{table\}\n?"):
    m = re.search(pat, b, re.S)
    moved.append(m.group(0))
    b = b.replace(m.group(0), "")

# wide tables / figures
b = b.replace("\\begin{table}[t]\n\\centering\n\\small\n\\caption{Learning 1{,}000", "\\begin{table*}[t]\n\\centering\n\\small\n\\caption{Learning 1{,}000")
b = b.replace("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n\\paragraph{Learning without forgetting.}", "\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n\\paragraph{Learning without forgetting.}")
b = b.replace("\\begin{figure}[t]", "\\begin{figure*}[t]").replace("\\end{figure}", "\\end{figure*}")
b = b.replace("0.48\\linewidth", "0.40\\textwidth")
b = b.replace("\\begin{table}[t]\n\\centering\n\\small", "\\begin{table}[t]\n\\centering\n\\footnotesize")
b = b.replace("\\begin{tabular}{llcccc}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llcccc}").replace(
    "\\bottomrule\n\\end{tabular}\n\\end{table}\n\nTable~\\ref{tab:removal}", "\\bottomrule\n\\end{tabular}}\n\\end{table}\n\nTable~\\ref{tab:removal}")
b = b.replace("\\multicolumn{6}{l}{Exact removal equal to never learned: \\textbf{yes} (v1 and v2, T4; v1 on CPU and in a 3-Dot smoke test)} \\\\\n"
              "\\multicolumn{6}{l}{Reset: value-table hash and output logits equal to the original model: \\textbf{yes}} \\\\",
              "\\multicolumn{6}{l}{Exact removal = never learned: \\textbf{yes} (v1, v2 on T4; v1 on CPU)} \\\\\n"
              "\\multicolumn{6}{l}{Reset: weights hash and logits equal the original: \\textbf{yes}} \\\\")

# anonymize + ACL-required sections
b = re.sub(r"\\section\{Limitations\}\n\\label\{sec:limits\}", r"\\section*{Limitations}\n\\label{sec:limits}", b)
b = b.replace("\\section{Broader Impact}", "\\section*{Ethical Considerations}")
b = re.sub(r"\\section\*\{Code and Data Availability\}.*?\n\n", "", b, flags=re.S)
b = re.sub(r"\\section\*\{Use of AI Assistance\}.*?\n\n",
           "\\\\section*{Use of AI Assistance}\nAn AI assistant helped with code, experiment scripts and drafting. The author verified the results and the text and is responsible for the content.\n\n", b, flags=re.S)
b = re.sub(r"\\section\*\{Author Contributions, Conflicts of Interest and Funding\}.*?\n\n", "", b, flags=re.S)
b = b.replace("\\section{Conclusion}", "\\section{Conclusion}", 1)
b = b.replace("is that the paper's code", "is that the code")

appendix = appendix.replace("\\section{Prompts}", "\\section{Prompts}")
extra = ("\\section{Algorithm and ablations}\n\\label{app:algo}\n"
         "Code, data subsets and result files are released in a public repository (link withheld for review).\n\n" +
         "\n".join(moved).replace("\\begin{table}[t]", "\\begin{table}[h]"))
tex = (r"\documentclass[11pt]{article}" "\n" r"\usepackage[review]{acl}" "\n"
       r"\usepackage{times}" "\n" r"\usepackage{latexsym}" "\n" r"\usepackage[T1]{fontenc}" "\n"
       r"\usepackage[utf8]{inputenc}" "\n" r"\usepackage{microtype}" "\n"
       r"\usepackage{amsmath,amssymb,amsthm}" "\n" r"\usepackage{booktabs}" "\n" r"\usepackage{graphicx}" "\n"
       r"\usepackage{xcolor}" "\n" r"\usepackage{algorithm}" "\n" r"\usepackage{algpseudocode}" "\n"
       r"\newtheorem{proposition}{Proposition}" "\n" r"\newcommand{\sys}{\textsc{Dots}}" "\n"
       rf"\title{{{TITLE}}}" "\n" r"\author{Anonymous ACL submission}" "\n"
       r"\begin{document}" "\n\\maketitle\n\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n" + b.rstrip() +
       "\n\n\\bibliography{references}\n\\bibliographystyle{acl_natbib}\n\n\\appendix\n" + extra + "\n" + appendix)
(HERE / "acl" / "main.tex").write_text(tex, encoding="utf-8")
print("written acl/main.tex; moved blocks:", len(moved))
