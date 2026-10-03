# Draft answers for the ARR submission form and Responsible NLP Research checklist

Always check the live form: wording and numbering change between cycles. Answer **yes / true** items only where they are true.

**A. Limitations.** Yes. Section "Limitations" (after the conclusion).
**A. Potential risks.** Yes. Section "Ethical Considerations": a model that keeps learning on a device can learn false or harmful material; poisoning risk; sharing learned units makes users distributors of their content.
**B. Artifacts used / created.**
- Used: Qwen2.5-0.5B (Apache-2.0), TriviaQA, NaturalQuestions-open, WikiText-2 (cited in the paper).
- Created: code, fact subsets and result files; released publicly after review under Apache-2.0.
- Intended use / license discussion: public benchmark datasets used for research as intended; no personal data collected.
- Documentation of artifacts: README and paper describe language (English), domain (trivia / Wikipedia) and size.
**C. Computational experiments.**
- Model sizes and parameters: Qwen2.5-0.5B; memory table 235M parameters; LoRA 17.6M; full fine-tuning 358M.
- Computing infrastructure and budget: free Colab NVIDIA T4 GPU; about 25 to 45 minutes per run; a 6-core laptop CPU for some runs.
- Hyperparameters: Appendix B.
- Descriptive statistics: **single runs only**; say this plainly (the paper does).
- Existing packages used: PyTorch, Hugging Face transformers, PEFT.
**D. Human annotators / data collection.** No new data collected; no human annotators. Answer "not applicable".
**E. Use of AI assistants.** **Yes.** An AI assistant (Claude, Anthropic) was used for code, experiment scripts and drafting. The author verified results and text and takes responsibility.

## Other form items
- Software and data upload: zip of `dots/`, `tests/`, `run_*.py`, `data/phase1_data.json`, `results/` with identifying links removed.
- Ethics: no human subjects; no IRB needed.
