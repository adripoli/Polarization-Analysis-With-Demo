# Live demo

One tweet goes through every version of the pipeline I built, side by side, so you can see each one get better
(or break something).

![demo](demo_screenshot.png)

## Run it

```bash
pip install -r requirements.txt        # main pipeline deps
pip install -r demo/requirements.txt   # + gradio for the web UI
python demo/app.py                     # open http://127.0.0.1:7860
```

- `python demo/app.py --cli "your tweet here"` prints every version's call in the terminal, no browser needed
- `python demo/app.py --share` also makes a temporary public link (lasts ~1 week) if someone wants to try it remotely

Works on CPU; a GPU just makes it snappier. Needs the two model files:
`model/bertweet_finetuned/best_model.pt` (final) and `model/mark4_rerun/mark4_seed1.pt` (Mark 4). They're ~540 MB
each, too big for GitHub, so they live outside git.

## The three tabs

1. **One tweet, every Mark.** Marks 1-3 are the actual old code from git (`demo/marks/`, copied unchanged from
   commits 1ba92a1, 22f45be and a2f78b5). Mark 4 is commit e554bad's model, retrained. Final is the saved model.
   Try "Those MAGA people are crazy" (the bug that started Mark 2) and the clown-car tweet (the regression Mark 2
   caused and Mark 3 fixed).
2. **Score a user.** Paste a few tweets, get the real polarization formula from `stage4_5_user_analysis.py` and see
   where that user would sit among the 30,525 users from the full run. The copy-paste bot example shows why Mark 4's
   ranking was broken.
3. **Grade it yourself.** Draws random tweets from the 169-tweet held-out test set (hand-labeled, never trained on)
   and keeps a running tally of which versions got it right.

## Known limitation

The saved final checkpoint's spam head collapsed (it says "spam" for everything), so the demo ignores it and only
uses the Biden-vs-Trump part, the same way the pipeline's stance score does. Real spam detection is next on the list.
