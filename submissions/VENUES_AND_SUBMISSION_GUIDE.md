# Where to submit the DOTS paper, and what to fill in

Files (in `submissions/`):
- `tmlr_submission.zip`: anonymous version for TMLR (12 pages)
- `ieee_submission.zip`: IEEE two-column version, for IEEE Access or IEEE Transactions (7 pages)
- `cover_letter.txt`: cover letter text for IEEE (and TMLR if asked)
- `DOTS_paper.pdf` (in `paper/`): the preprint version

## Venues, ranked for this paper
Rule for all of them: **submit to one venue at a time.** Submitting the same paper to two venues at once is a dual-submission violation and gets both rejected. A preprint on arXiv or Zenodo is allowed at all of the venues below.

| # | Venue | Recognition | Cost | Speed | Fit for this paper |
|---|---|---|---|---|---|
| 1 | **TMLR** (Transactions on Machine Learning Research), openreview.net/group?id=TMLR | Well respected in ML. Papers can earn "Featured" and "Reproducibility" certifications and can be presented at ICLR. | Free | Pre-screen about 2 weeks, full review 2 months | **Best fit.** It judges correctness and evidence, not novelty or scale, which suits a first, small-scale result. |
| 2 | **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**, mc.manuscriptcentral.com/tnnls | Highest-prestige IEEE option in this area | Free to submit (optional page charges) | About 3 to 5 months | Reviewers will likely ask for larger models and more datasets. Worth it after Phase 2 experiments. |
| 3 | **IEEE Transactions on Artificial Intelligence (TAI)**, IEEE Computational Intelligence Society | IEEE-branded | Free to submit (optional charges) | Similar to TNNLS | Plausible alternative to TNNLS. |
| 4 | **IEEE Access**, ieee.org/ieee-access | IEEE-branded mega-journal, lower selectivity | Open-access fee, about $2,160 in 2026 (check current fee) | First decision about 30 days | Fast and likely to be accepted, but lower prestige and not free. |
| 5 | **ACL Rolling Review → NAACL 2027 / COLING 2027** | Strong in NLP | Free | ARR deadline **12 Oct 2026**, decisions Dec 2026 | Needs the ACL template, a Limitations section (included) and the Responsible NLP checklist. The paper must be recast for an NLP audience. Tight: 9 days. |
| 6 | **ICML 2027 / NeurIPS 2027 main track** | Top | Free | Deadlines around Jan and May 2027 (unverified; check each site) | Needs stronger evidence: 1B to 8B models, more datasets, skill tasks. The ICLR 2027 deadline has already passed (25 Sep 2026). |
| 7 | **Workshops** (NeurIPS / ICML / ICLR workshops on unlearning, continual learning, memory) | Moderate, quick recognition | Free | 4 to 6 page papers | Good for early feedback. Many allow non-archival submissions. |

My recommendation: **submit to TMLR now.** In parallel, run bigger-model experiments (Phase 2) so a stronger version can go to TNNLS or a 2027 conference.

## TMLR: what to fill in (OpenReview)
1. Create an OpenReview account with your email and complete your profile (name, affiliation, DBLP/Google Scholar links if any). OpenReview moderates new profiles, which can take days: do this first.
2. Go to openreview.net/group?id=TMLR → **Submit**.
3. Fields:

| Field | Enter |
|---|---|
| Title | `DOTS: Exactly Removable Continual Learning for Frozen Language Models via Hyperdimensional Sparse Memory` |
| Authors | Sohan Merugu (your OpenReview profile) |
| Abstract | text from `abstract_plain.txt` |
| PDF | `tmlr/build/main.pdf` (inside the zip as `main.pdf`). It is anonymous. |
| Supplementary material (optional) | a zip of the code. Remove your name and any GitHub link first. |
| Code URL | leave blank or give an anonymized link; the public repo reveals your name |
| Previous venue / dual submission | None |
| Competing interests | None |
| Preprint | If asked about arXiv/Zenodo, say yes if you posted one. This is allowed. |

4. TMLR's PDF is already anonymous: no author name, no GitHub link. After acceptance you switch the style to the accepted version and add the real author and links.

## IEEE Access: what to fill in (ScholarOne)
1. Create an account at ieee.org (free) and at ieeeaccess.ieee.org → **Submit**.
2. Use the official template from ieee.org/ieee-access (Word or LaTeX). Our `ieee/main.tex` uses the standard IEEEtran class; copy the text and figures into the Access template or ask me to convert it.
3. Fields:

| Field | Enter |
|---|---|
| Title | same as above |
| Abstract | `abstract_plain.txt` (about 220 words, IEEE style: no citations or equations) |
| Keywords | continual learning; machine unlearning; sparse memory; large language models; catastrophic forgetting; hyperdimensional computing |
| Manuscript type | Regular paper |
| Author | Sohan Merugu, Independent Researcher, sohanmerugu@gmail.com. Add an ORCID (free at orcid.org) |
| Funding | None |
| Cover letter | upload `cover_letter.txt` |
| Suggested reviewers (3 to 5) | authors of the closest work: Jessy Lin (sparse memory finetuning), Hadi Pouransari (hierarchical memories), Tianyu Zhao (fast-weight PKM), Aditi Raghunathan (exact unlearning). Do not suggest people you know personally. |
| Data availability | `https://github.com/sohan-a11y/open-dots` |
| Previous publication | arXiv/Zenodo preprint, if posted |

4. Payment: IEEE Access charges the open-access fee after acceptance.

## IEEE TNNLS / TAI: what to fill in
Same as IEEE Access, through ScholarOne (mc.manuscriptcentral.com/tnnls). TNNLS asks for: article type (Regular Paper), a 150 to 250 word abstract, 3 to 5 suggested reviewers, whether the work was posted as a preprint, and confirmation that the paper is not under review elsewhere. Do not submit here until the larger-scale experiments are done.

## Before any submission
- Create an ORCID (orcid.org, free): most journals ask for it.
- State the AI use honestly. The paper includes an "Use of AI Assistance" section in the preprint and IEEE versions. TMLR's anonymous version omits it from the body; declare it in the submission form if asked.
- Read the full PDF; the author is responsible for every claim.
