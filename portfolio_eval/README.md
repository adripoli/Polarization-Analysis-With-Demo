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
6. `stageA_clean.py`, then `stageC_light.py` (bot/non-English counts, no model) and `stageB_score_v2.py` + `stageC_users.py` (scores all 1.4M unique tweets in chunks, then the top-100 check). `stageB_score.py` is the first version that ran out of RAM; kept for the record
6b. `probe_side.py`: the 12 fresh tweets that show how the models do outside their training slice
7. `figs.py`, `fig_users.py`, `build_report.py`, `build_short.py`: charts and both PDFs. `record_demo.py` + `make_video.py`: the demo video (needs `pip install playwright` + `python -m playwright install chromium`, and ffmpeg)

8. Mark 7, run from the project root after `python main.py`:
   - `eval_mark7.py`: Mark 4, old final and Mark 7 on the held-out, supportive-tweet and fresh-tweet tests
   - `audit_mark7.py`: hunts for test tweets hiding in the training data, then draws the blind sample
   - `blind_test.py`: grades the random untagged tweets
   - `user_check.py`: user scores vs the side people declared with hashtags
   - `mark7_seeds.py`: retrains on new seeds so one lucky run can't fool anyone

Heads up: `model/bertweet_finetuned/best_model.pt` predicts Spam for every tweet (collapsed run). See section 7 of the report.
