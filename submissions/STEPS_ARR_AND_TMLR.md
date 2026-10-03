# Submitting to ACL Rolling Review (ARR) and TMLR

Submit to **one at a time** (dual submission is not allowed). Order I suggest: ARR on or before **12 Oct 2026** (hard deadline), then TMLR if you want a second route, but only after the ARR result or once you withdraw. Check ARR's current rules on arXiv/preprints and on withdrawing before you decide.

Files:
- `acl_submission.zip` (ARR): anonymous, ACL template, main text about 6.5 pages, Limitations and Ethical Considerations sections included
- `tmlr_submission.zip` (TMLR): anonymous, TMLR template, 12 pages
- `ARR_checklist_answers.md`: draft answers for the ARR submission form and Responsible NLP checklist
- `abstract_plain.txt`: abstract text for the forms

## Step 0: do first (can take days)
1. Create an **OpenReview** account at openreview.net with your email.
2. Fill your profile **completely**: name, institution ("Independent Researcher"), emails, homepage/GitHub, DBLP or Google Scholar link if you have one. Incomplete profiles are blocked from submitting. Moderation of new profiles can take several days: do this today.
3. Create an **ORCID** at orcid.org (free) and add it to the OpenReview profile.

## ARR (ACL Rolling Review), October 2026 cycle
1. Go to aclrollingreview.org → Submit (OpenReview). Deadline **12 October 2026**. Check the time zone (AoE) on the site.
2. Submission form fields:

| Field | Enter |
|---|---|
| Title | `DOTS: Exactly Removable Continual Learning for Frozen Language Models via Hyperdimensional Sparse Memory` |
| Authors | Sohan Merugu (via your OpenReview profile) |
| Abstract | `abstract_plain.txt` |
| PDF | `main.pdf` from `acl_submission.zip` (anonymous) |
| Paper type | Long paper (8 pages) |
| Research area / track | Interpretability and Analysis of Models for NLP, or Language Modeling, or Machine Learning for NLP. Pick the closest of those |
| Keywords | continual learning; machine unlearning; knowledge editing; sparse memory; language models |
| Software / data | Upload a zip of the code, with your name and the GitHub link removed |
| Preprint | State whether a preprint exists |
| Submit to a venue | ARR cycle first; you later commit it to a conference (NAACL 2027 or COLING 2027, commitment deadline 20 Dec 2026) |
| Reviewer registration / reviewing commitment | ARR requires an author to register as a reviewer for the cycle. Check the CFP for the exact rule and fill it in with your profile |
| AI assistance | Answer honestly in the Responsible NLP checklist: yes, an AI assistant helped with code and drafting (see `ARR_checklist_answers.md`) |

3. The ARR format limits main text to 8 pages. Limitations is required and is included. References and appendices don't count.

## TMLR
1. Go to openreview.net/group?id=TMLR → **Submit**.
2. Fields:

| Field | Enter |
|---|---|
| Title / Abstract | as above |
| PDF | `main.pdf` from `tmlr_submission.zip` (anonymous) |
| Supplementary (optional) | anonymized code zip |
| Code URL | leave blank (the public repo reveals your name) |
| Competing interests | None |
| Dual submission | confirm it is not under review anywhere else |

3. TMLR pre-screens in about 2 weeks, then reviews for about 2 months. It checks correctness of claims, not novelty.

## AI-use statement
The paper includes one neutral sentence: an AI assistant helped with code, scripts and drafting, and the author verified and is responsible for the content. This is accurate. IEEE policy asks for such disclosure, and the ARR checklist asks directly. Do not change the answers to "no": reviewers and editors judge the experiments, and a false answer is grounds for rejection or retraction.
