# portfolio_eval

Everything used to re-grade every old version of the pipeline for the MIT portfolio report
(`Polarization Pipeline MIT Maker Portfolio.pdf` on the desktop).

The scripts were run from a scratch folder. They take paths as arguments, and the old commits come from
`git archive <hash>` of the GitHub repo. Run order:

1. `make_split.py <outdir>`: rebuilds the final model's train/val split, writes `test_heldout.csv` (169 tweets it never trained on)
2. `cache_raw.py <outdir>/raw.pkl`: loads the 1.75M raw tweets once
3. `eval_keyword.py <commit_dir> <outdir>`: runs one Mark 1-3 commit's `score_tweet()` unchanged, gives `kw_<hash>.json`
4. `eval_mark4.py <commit_dir> <outdir> full|frozen <seed>`: retrains e554bad / bd4d639 with their own code, gives `m4_*.json`
5. `eval_final.py <outdir>`: grades `model/bertweet_finetuned/best_model.pt` on the held-out set, gives `final_eval.json`
6. `stageA_clean.py` then `stageC_light.py`: Mark 5/6 cleaning + bot/non-English user counts (`stageB_score.py` / `stageC_users.py` = the model-based version, stopped early on low memory)
7. `figs.py`, `fig_users.py`, `build_report.py`: charts and the PDF

Heads up: the saved `best_model.pt` predicts Spam for every tweet (collapsed run). See section 7 of the report.
