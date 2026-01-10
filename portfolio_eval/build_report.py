# build_report py turns
import json, os, glob
import numpy as np, pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
                                Table, TableStyle, PageBreak, KeepTogether, CondPageBreak)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib

W = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(W, "..", "figs")
OUT = os.environ.get("OUT") or os.path.join(os.path.expanduser("~"), "Desktop", "Polarization Pipeline MIT Maker Portfolio.pdf")

# fonts dejavu has
T = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for name, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(name, os.path.join(T, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#8a8984")
BLUE, ORANGE, AQUA = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")
PAPER, RULE, TINT, WARN = colors.HexColor("#fcfcfb"), colors.HexColor("#e4e3df"), colors.HexColor("#f3f2ee"), colors.HexColor("#fdf3ec")

S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14.2, textColor=INK, spaceAfter=6),
    "small": ParagraphStyle("small", fontName="DV", fontSize=8.2, leading=11.2, textColor=INK2, spaceAfter=4),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.2, leading=11.2, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=INK, spaceBefore=6, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=16, textColor=INK, spaceBefore=10, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=9.8, leading=13, textColor=BLUE, spaceBefore=6, spaceAfter=2),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.6, leading=14, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.2, leading=10.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.2, leading=10.6, textColor=INK),
    "mono": ParagraphStyle("mono", fontName="DVM", fontSize=7.4, leading=9.6, textColor=INK),
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=30, leading=34, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12.5, leading=17, textColor=INK2, spaceAfter=14),
    "tile_n": ParagraphStyle("tn", fontName="DV-B", fontSize=15, leading=18, textColor=INK, alignment=TA_LEFT),
    "tile_l": ParagraphStyle("tl", fontName="DV", fontSize=7.8, leading=10, textColor=INK2, alignment=TA_LEFT),
}
CONTENT_W = letter[0] - 1.5 * inch


def P(t, st="body"):
    return Paragraph(t, S[st])


def B(items):
    return [Paragraph(i, S["bul"], bulletText="•") for i in items]


def fig(name, cap, w=CONTENT_W):
    p = os.path.join(FIG, name)
    from PIL import Image as PI
    iw, ih = PI.open(p).size
    return KeepTogether([Image(p, width=w, height=w * ih / iw), Spacer(1, 3), P(cap, "cap")])


def box(flow, bg=TINT, border=None):
    t = Table([[flow]], colWidths=[CONTENT_W])
    st = [("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 10),
          ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 8),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]
    if border:
        st.append(("LINEBEFORE", (0, 0), (0, -1), 3, border))
    t.setStyle(TableStyle(st))
    return t


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else P(c, "cellb" if (header and i == 0) else "cell") for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
          ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE)]
    if header:
        st += [("LINEBELOW", (0, 0), (-1, 0), 1, INK2)]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), TINT))
    t.setStyle(TableStyle(st))
    return t


def tiles(items):
    cells = [[P(n, "tile_n"), P(l, "tile_l")] for n, l in items]
    inner = [Table([[a], [b]], colWidths=[CONTENT_W / len(items) - 8]) for a, b in cells]
    for t in inner:
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 2),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    t = Table([inner], colWidths=[CONTENT_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (1, 0), (-1, 0), 2, PAPER),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    return t


# load all the
kw = {os.path.basename(f)[3:-5]: json.load(open(f)) for f in glob.glob(os.path.join(W, "kw_*.json"))}
m4 = np.array([json.load(open(f))["pol_acc"] for f in sorted(glob.glob(os.path.join(W, "m4_full_*.json")))])
m4f = np.array([json.load(open(f))["pol_acc"] for f in sorted(glob.glob(os.path.join(W, "m4_frozen_*.json")))])
fin = json.load(open(os.path.join(W, "final_eval.json")))
U = json.load(open(os.path.join(W, "users_light.json")))
T = json.load(open(os.path.join(W, "users_eval.json")))["results"]
pct = lambda x, d=0: f"{x*100:.{d}f}%"
K1, K2, K3 = kw["1ba92a1"], kw["22f45be"], kw["a2f78b5"]


def on_page(c, doc):
    c.saveState()
    c.setFillColor(PAPER); c.rect(0, 0, letter[0], letter[1], stroke=0, fill=1)
    if doc.page > 1:
        c.setFont("DV", 7.5); c.setFillColor(MUTED)
        c.drawString(0.75 * inch, 0.5 * inch, "Polarization Pipeline · MIT Maker Portfolio reference notes")
        c.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, str(doc.page))
        c.setStrokeColor(RULE); c.setLineWidth(0.5)
        c.line(0.75 * inch, 0.65 * inch, letter[0] - 0.75 * inch, 0.65 * inch)
    c.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                      topMargin=0.7 * inch, bottomMargin=0.85 * inch,
                      title="Polarization Pipeline MIT Maker Portfolio", author="Adrian Polishchuk Carrillo")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")],
                                   onPage=on_page)])
s = []

# cover
s += [Spacer(1, 0.35 * inch), P("MIT MAKER PORTFOLIO · REFERENCE NOTES", "small"), Spacer(1, 4),
      P("Polarization Pipeline", "title"),
      P("Measuring how split Twitter users were during the 2020 US election, and the six-plus versions "
        "it took to get a number I could actually trust.", "sub"),
      tiles([("1.75M", "tweets from #trump and #joebiden"), ("46", "commits, Oct 10 to Dec 15, 2025"),
             ("7", "versions: Marks 1–6 + final"), ("8→80%", "held-out accuracy, Mark 1 vs final"),
             ("30,525", "users scored in the final run")]),
      Spacer(1, 14),
      box([P("<b>Read this first</b>", "body"),
           P("This is reference material for writing your own portfolio entry, not the entry itself. It's written "
             "casually on purpose so you can lift ideas without it sounding like a textbook. ", "body"),
           P("Every number in here was measured for this report: each old version was pulled back out of git and "
             "rerun on the same test set (section 3 explains how). A few things in your original notes don't "
             "match what git shows. Those are listed in section 8, so fix them before they go into the real thing.", "body")],
          bg=TINT, border=BLUE),
      Spacer(1, 14),
      table([["", ""],
             ["Maker", "Adrian Polishchuk Carrillo"],
             ["Code", "github.com/adripoli/Polarization-on-Social-Media"],
             ["Built with", "Python, PyTorch, Hugging Face Transformers (BERTweet), scikit-learn, pandas, langdetect"],
             ["Hardware", "RTX 3070, about 5 hours for one full end-to-end run"],
             ["Data", "~1.75M election tweets in two CSVs: #trump (971,157) and #joebiden (777,078), "
                      "plus 1,000 tweets I labeled by hand"]],
            [1.1 * inch, CONTENT_W - 1.1 * inch], header=False, zebra=False),
      Spacer(1, 10),
      P("<b>Contents</b>", "h3"),
      P("1. The project in a paragraph  ·  2. What I built  ·  3. How every version was retested "
        " ·  4. Timeline  ·  5. Mark by mark  ·  6. Final pipeline and results  ·  6b. Mark 7 "
        " ·  7. Things found while rerunning everything  ·  8. Fact-check of the original notes "
        " ·  9. Mark ? (what's next)  ·  10. Engineering lessons  ·  "
        "11. Using this for MIT  ·  12. Demo and codebase  ·  Appendix: every commit, test details", "small"),
      PageBreak()]

# mark 7 numbers
_R = r"C:\Users\Asus Gaming\Desktop\political_polarization\portfolio_eval"
_j = lambda n: json.load(open(os.path.join(_R, n)))
E7, BL7, UC7, SD7 = _j("mark7_eval.json"), _j("blind_eval.json"), _j("user_check.json"), _j("mark7_seeds.json")
H7 = np.array([E7["heldout"]["Mark 7"]["acc"]] + [v["heldout"] for v in SD7.values()])
ALL7 = {k: [v[k] for v in SD7.values()] for k in ("heldout", "shortcut", "fresh", "blind")}
pc0 = lambda x: f"{x*100:.0f}%"

# 1 one paragraph
s += [P("1. The project in a paragraph", "h1"),
      P("I wanted to measure political polarization on Twitter. Classifying single tweets as pro-Trump or pro-Biden "
        "wasn't enough; I wanted to know whether each <i>person</i> stayed on one side, and whether the whole "
        "population split into two camps. The data was about 1.75 million tweets from the 2020 US election. I built "
        "it as a pipeline: tweets go in, every tweet gets a stance score, every user gets a polarization score, and "
        "at the end there's a population-level measure of how far apart the two sides are."),
      P("The first three versions were hand-written rules (keywords, then adjectives, then word distance). They all "
        "hit the same wall, so I switched to fine-tuning a transformer language model (BERTweet) on 1,000 tweets I "
        "labeled by hand. The versions after that were about fixing what the model got wrong at the user level: "
        "spam accounts, non-English tweets, and a scoring formula that rewarded copy-paste bots. Then a live demo showed "
        "the models had learned a shortcut, which led to Mark 7."),
      fig("fig2_accuracy.png",
          f"Figure 1. The whole project in one chart. Every version graded on the same {155} hand-labeled political tweets "
          "that the final model never trained on. The rule-based versions are right when they make a call, but they "
          "skip almost 9 out of 10 tweets. Mark 4 is the mean of retraining 5 times, Mark 7 of 2 (± = standard deviation)."),
      ]

# 2 what I
s += [P("2. What I built", "h1"),
      P("Everything through the final version is my code except the pretrained BERTweet weights and the Python libraries. The pieces:"),
      table([["Part", "What it does", "File"],
             ["Hand-labeled dataset", "1,000 tweets I read one by one and labeled pro-Biden, pro-Trump or spam. Started from an "
              "auto-labeled sheet (by hashtag) and corrected it by hand. Used for training and for grading every version.",
              "data/manual_labels.csv"],
             ["Rule-based scorers (Marks 1-3)", "Keyword lists, adjective flipping, word-distance weighting.",
              "analyze_tweets.py, keywords.py, adjectives.py"],
             ["Stance classifier", "BERTweet (12-layer transformer pretrained on 850M tweets) with a small head I added on top "
              "(768 → 256 → 3 classes). Outputs P(Biden), P(Trump), P(spam) and a stance from −1 to +1.", "model.py"],
             ["Training loop", "Weak labels from the hashtag file + hand labels counted 5×, stratified split, class weights, "
              "label smoothing, early stopping, mixed precision.", "stage3_training.py"],
             ["Cleaning", "Strips URLs, normalizes @mentions, turns emoji into text, drops retweets/duplicates/non-English. "
              "Cached, so reruns skip the slow language detection.", "stage0_cleaning.py"],
             ["User scoring", "Per-user mean and spread of stance, model confidence, consistency and spam ratio, combined "
              "into one polarization score.", "stage4_5_user_analysis.py"],
             ["Population analysis", "Histogram of user stances, 2-cluster Gaussian mixture, overlap between clusters, "
              "polarization vs activity.", "stage6_analysis.py"],
             ["Orchestration", "One command runs it all; <font name='DVM'>--skip-training</font> reuses a saved model.", "main.py, config.py"]],
            [1.35 * inch, CONTENT_W - 1.35 * inch - 1.55 * inch, 1.55 * inch]),
      ]

# 3 how retested
s += [P("3. How every version was retested", "h1"),
      P("Your notes say what each Mark did and what broke, but not <i>how well</i> each one worked. So for this report "
        "every old version came back out of git and got graded on the same test. That kind of evidence makes a "
        "portfolio entry stronger, and you can rerun it yourself (scripts in Appendix B)."),
      *B([f"<b>The test set.</b> 169 hand-labeled tweets (155 political, 14 spam) that the final model never saw in training. "
          "They were found by rebuilding the final model's exact train/validation split (same data, same seed 42). Sanity check: "
          f"the final model gets {pct(fin['train_manual']['pol_acc'])} on hand-labeled tweets it trained on and "
          f"{pct(fin['test_heldout']['pol_acc'])} on these 169. That gap is what a real held-out set looks like.",
          "<b>Rule-based Marks 1-3.</b> Each commit's own <font name='DVM'>score_tweet()</font> was run with zero "
          "changes, 11 commits in total. They don't train, so they were also run on all 803 political hand-labeled "
          "tweets and on all 1.75M raw tweets.",
          "<b>Mark 4.</b> Retrained from commit e554bad's own code and settings (3 epochs, learning rate 2e-5, batch 32), "
          "five times with different random seeds, so the number isn't a lucky run. Same for the frozen-encoder "
          "attempt (bd4d639), three seeds.",
          "<b>Marks 5 and 6.</b> These changed the cleaning, not the model, so tweet-level accuracy doesn't move. "
          "They were meant to fix the user rankings, so that's what gets measured: how many users are guaranteed a "
          "fake “perfect” zero spread (every tweet identical), and how many post mostly in other languages, "
          "under each version's cleaning. Neither needs the model, so it's exact.",
          "<b>What “correct” means.</b> The predicted side matches the hand label. A 0 (no call) counts as wrong, "
          "because the tweet did have a side and the program missed it. \u201cRight when it made a call\u201d is also reported "
          "separately so the rule-based versions get a fair shake."]),
      ]

# 4 timeline
s += [CondPageBreak(4 * inch), P("4. Timeline", "h1"),
      P("Marks 1 to 3 happened in four days. The switch to machine learning took six weeks, most of it on labeling data "
        "and getting training to work. The last three rounds were quick, targeted fixes, each one triggered by "
        "looking at the output and finding something wrong."),
      fig("fig5_timeline.png", "Figure 2. Every development commit, grouped by Mark. (3 later commits that only add the "
          "license and polish the readme are left out.)"),
      fig("fig1_pipeline.png", "Figure 3. How the pipeline grew. Each row is one version; the colored box is what that "
          "version added. Mark 4 threw out the rule stages entirely instead of stacking another rule on top."),
      ]

# 5 mark by
def mark(title, when, built, wrong, nums, nxt):
    out = [CondPageBreak(2.2 * inch), P(title, "h2"), P(when, "small")]
    out += [P("What I built", "h3")] + (built if isinstance(built, list) else [P(built)])
    out += [P("What went wrong", "h3")] + (wrong if isinstance(wrong, list) else [P(wrong)])
    out += [box(nums if isinstance(nums, list) else [P(nums)], bg=TINT, border=BLUE)]
    out += [P("What I changed next", "h3")] + (nxt if isinstance(nxt, list) else [P(nxt)])
    return out


t1, t2, t3 = K1["test_heldout"], K2["test_heldout"], K3["test_heldout"]
a1, a2, a3 = K1["manual_all_dedup"], K2["manual_all_dedup"], K3["manual_all_dedup"]
s += [PageBreak(), P("5. Mark by mark", "h1"),
      P("Same structure for every version: what I built, what went wrong, the measured numbers (blue boxes), "
        "and how I adapted. That last part is the engineering-process story MIT readers are looking for.")]

s += mark("Mark 1: keyword matching", "Oct 10 · commits da08515 → 705d7d1 · 7 commits",
          "Skimming the data, pro-Trump tweets kept using the same hashtags (#VoteTrump, #Trump2020, #MAGA) and pro-Biden "
          "tweets had their own (#VoteBiden, #BidenHarris2020). So Mark 1 checks every tweet for one of 48 keywords "
          "(a mix of mine and AI-suggested ones) and scores it +1 for Trump, −1 for Biden, or 0 if it has neither "
          "or both. Then it groups by user and looks at each person's average and spread.",
          [P(f"Two big problems. First, most tweets don't contain any of the keywords: rerun on all 1.75M tweets, "
             f"<b>{pct(K1['full']['neutral'])} came out 0</b>. Second, a keyword says who a tweet is about, not which side "
             "it's on. “Those MAGA people are crazy” scores +1 (pro-Trump) because it has “maga” in it."),
           P("Rerunning it also turned up sneakier bugs from plain substring matching. The keyword “45” (as in "
             "45th president) matches any tweet with the digits 45 in it, a time, a vote count, anything. That alone tagged "
             "<b>10,080 tweets</b> as pro-Trump for no reason. And “maga” matches inside “magazine” "
             "(767 tweets).")],
          [P(f"<b>Rerun on 155 held-out political tweets:</b> {pct(t1['accuracy'])} correct, {pct(t1['wrong_side'])} wrong side, "
             f"{pct(1-t1['coverage'])} no call. When it did make a call it was right {pct(t1['precision_called'])} of the time "
             f"({pct(a1['precision_called'])} on all 803 labeled tweets). So the keywords weren't useless; they just "
             "almost never fired.")],
          "The mocking-MAGA tweet was the error I could actually see in the output, so that's what I went after first. "
          "(Looking back, the 82% no-call problem was much bigger, but I hadn't measured it yet. More on that in section 10.)")

s += mark("Mark 2: adjectives flip the score", "Oct 10 · commits 112781e → 22f45be · 4 commits",
          [P("Added lists of positive and negative adjectives. If a tweet has a keyword and a negative adjective, flip "
             "the score. “Those MAGA people are crazy” → “crazy” is negative → flip to "
             "pro-Biden. Fixed."),
           P("Then testing on more tweets turned up “trump is crazy good at rallies,” where crazy is a compliment "
             "(commit 5fd63dd). Patch: only flip when there's a negative word and <i>no</i> positive word (05dec46).")],
          [P("Tweets with several adjectives aimed at different people. “Incompetent. Just like everyone else involved "
             "in the trump clown car. VoteBiden”: Mark 1 actually got this one <i>right</i> (VoteBiden → Biden). "
             "Mark 2 sees “incompetent,” flips it, and gets it wrong. So Mark 2 fixed one case and broke another. "
             "That's a regression, and it's a great detail for the write-up.")],
          [P(f"<b>Rerun:</b> {pct(t2['accuracy'])} correct on held-out, exactly the same as Mark 1. On all 803: "
             f"{pct(a2['accuracy'],1)} vs {pct(a1['accuracy'],1)}. Why so flat? Only <b>3.1% of the hand-labeled tweets "
             "(27 of 874)</b> contain both a keyword and an adjective, so the flip rule almost never fires. The fix was "
             "aimed at a real error that hardly ever happens.")],
          "Fix the multiple-adjectives problem by figuring out which keyword each adjective is actually describing.")

s += mark("Mark 3: weight adjectives by distance", "Oct 11–13 · commits 9f440ef → a2f78b5 · 6 commits",
          [P("Split the tweet into words. For each keyword, find the nearest adjective, weight it by 1 / (1 + distance), and "
             "ignore anything more than 6 words away. Each keyword gets its own vote instead of one flag per tweet. Along the "
             "way: a divide-by-zero when the adjective sat right next to the keyword (fixed in 0c80a29), the distance cap "
             "(6970d94), and one vote per keyword (a37cba4)."),
           P("The clown-car tweet comes out right again, because “incompetent” is far away from “VoteBiden.”")],
          [P(f"Overall it didn't help and slightly hurt: coverage dropped from {pct(a2['coverage'])} to {pct(a3['coverage'])}. "
             "Splitting into single words means the five two-word keywords (“trump train,” “vote blue,” "
             "“biden won”…) can never match anymore. Rerunning also showed Mark 3 <b>crashes</b> on the full "
             "dataset: there are 167 empty tweets in the CSVs and the new tokenizer dropped the <font name='DVM'>str()</font> "
             "cast the earlier versions had."),
           P("And sarcasm is basically impossible with rules. “Ugh, of course the MAGA crowd thinks this is brilliant. "
             "Great job guys. VoteBiden” gets no call from every rule-based version.")],
          [P(f"<b>Rerun:</b> {pct(t3['accuracy'])} correct on held-out, {pct(a3['accuracy'])} on all 803 (Mark 2 was "
             f"{pct(a2['accuracy'])}). Right-when-it-calls stayed around {pct(a3['precision_called'])}. Three versions, "
             "11 commits, and the overall number moved by about one point (Figure 4).")],
          "This is where I stopped patching rules. Every fix added another special case and the number didn't budge. Rules "
          "only see words; they can't see meaning, sarcasm or grammar. Time for a model that learns from examples.")

cases = [["Test sentence", "Hand label", "Mark 1", "Mark 2", "Mark 3"],
         ["Those MAGA people are crazy", "Biden", "Trump ✗", "Biden ✓", "Biden ✓"],
         ["Incompetent. Just like everyone else involved in the trump clown car. VoteBiden", "Biden", "Biden ✓",
          "Trump ✗", "Biden ✓"],
         ["trump is crazy good at rallies", "Trump", "no call", "no call", "no call"],
         ["Ugh, of course the MAGA crowd thinks this is brilliant. Great job guys. VoteBiden", "Biden", "no call",
          "no call", "no call"],
         ["MAGA! Trump 2020, four more years, the best president ever", "Trump", "Trump ✓", "Trump ✓", "Trump ✓"]]
s += [Spacer(1, 6), KeepTogether([P("Case study: the same sentences through each rule-based version", "h3"),
      table(cases, [CONTENT_W - 4 * 0.95 * inch] + [0.95 * inch] * 4),
      P("Run through each commit's actual code. Row 1 is the bug that started Mark 2; row 2 is the regression Mark 2 caused "
        "and Mark 3 fixed. Row 3 gets no call because plain “trump” isn't a keyword, only the hashtag-style ones "
        "are. Row 4 (sarcasm) is the kind of tweet that pushed me to ML.", "cap")]),
      fig("fig3_commits.png", "Figure 4. Every rule-based commit rerun on all 803 hand-labeled political tweets. Precision "
          "(purple) was fine the whole time. Coverage (green) was the real bottleneck, and no rule change touched it.")]

frz, ful = m4f.mean(), m4.mean()
s += mark("Mark 4: fine-tune BERTweet", "Oct 19 – Nov 30 · commits 5343537 → 07422d9 · 9 commits",
          [*B(["<b>Spike before rewriting.</b> Before throwing out the rule code, I tested that BERTweet's tokenizer handles "
               "hashtags, emoji and slang on a few sample tweets (5343537). Only after that did I commit to the rewrite.",
               "<b>Labels.</b> A model needs examples. <font name='DVM'>create_manual_labels.py</font> dumped 1,000 tweets "
               "auto-labeled by which hashtag file they came from, then I went through every one by hand (7f428f9). "
               "The auto-labels were wrong most of the time: <b>only 29% of the political tweets matched my hand label</b>. "
               "The first 500 tweets of the #joebiden file are mostly attacks on Biden (Hunter Biden emails, Burisma), "
               "and the #trump file is mostly anti-Trump. A hashtag tells you who a tweet is about, not which side it's on. "
               "About 90 were spam or promo and got their own label.",
               "<b>A silent tokenizer bug.</b> BERTweet doesn't ship a “fast” tokenizer, and the default quietly "
               "fell back to one that split hashtags and emoji wrong without any error. Forcing "
               "<font name='DVM'>use_fast=False</font> fixed it (dd38be1).",
               "<b>Training.</b> First try: freeze BERTweet and only train the small head on top (bd4d639). Then unfroze "
               "everything at a small learning rate (f6a4f87). The loss then dropped to near zero within an epoch, which "
               "meant it was memorizing ~800 tweets, so I added an 80/20 split to watch for overfitting and more dropout."])],
          [P("Ranking users by lowest spread (“most consistent = most polarized”) put spam accounts at the top: the "
             "same promo tweet posted over and over gives zero spread, which looks like a rock-solid opinion (noted in "
             f"07422d9). Measured on the full data: of the {U['Mark 4']['users_2plus']:,} users with 2+ tweets, "
             f"<b>{U['Mark 4']['all_identical_users']:,} ({pct(U['Mark 4']['all_identical_share'])})</b> only ever posted "
             "one tweet, over and over. Every one of them gets a spread of exactly zero whatever the model says, so they "
             "all tie for first place on the “most polarized” list. Scoring all 1.75M tweets with the Mark 4 model "
             f"and ranking like Mark 4 did: <b>{pct(T['Mark 4']['top_copy_paste'])} of the top 100</b> were copy-paste "
             f"accounts, and {pct(T['Mark 4']['top_non_english'])} were mostly non-English.")],
          [P(f"<b>Rerun on 155 held-out political tweets:</b> <b>{pct(ful)} ± {m4.std()*100:.1f}</b> correct (\u00b1 = points of spread) across 5 "
             f"retrains, with a call on every tweet. Frozen encoder: {pct(frz)}. Both are way above the 53% you'd get by "
             "always guessing Trump. From ~9% to ~76% is the biggest single jump in the project."),
           P("<b>Heads up on the frozen version:</b> the bd4d639 commit message says the loss sat at ln 2 (coin-flip level) no "
             "matter what. In the rerun it didn't: loss came down to about 0.5–0.6 and it got 70% right (Figure 5). "
             "Maybe different library versions or settings back then. Fine-tuning still wins by ~6 points, but don't "
             "write that the frozen version was no better than random.", "small")],
          "Get the spam accounts out of the rankings before the ranking means anything.")
s += [fig("fig4_loss.png", "Figure 5. Training loss from the reruns. Thick line = first seed, faint lines = the other seeds. "
          "Both versions learn; the fine-tuned one learns more smoothly and ends up more accurate on held-out tweets.",
          w=CONTENT_W * 0.82)]

u4, u5, u6 = U["Mark 4"], U["Mark 5"], U["Mark 6"]
t4, t5, t6 = T["Mark 4"], T["Mark 5"], T["Mark 6"]
s += mark("Mark 5: clean out short and repeated tweets", "Dec 2–5 · commits 1503a65 → 27dc538 · 4 commits",
          [*B(["<b>Try 1:</b> drop tweets under 20 characters (1503a65). Too crude. Real short tweets got dropped, and spam "
               "that padded itself with hashtags slipped through.",
               "<b>Try 2:</b> count words instead of characters, keep tweets with more than 5 (e342338). Caught an "
               "off-by-one before committing: <font name='DVM'>&gt;=</font> let a pile of 5-word boilerplate through, so it "
               "became a strict <font name='DVM'>&gt;</font>.",
               "<b>Try 3:</b> the word filter alone still left lots of spam, because most of it was the exact same tweet from "
               "the same account. Added a dedupe on (user, text) (31261c1), and moved cleaning to run <i>before</i> the slow "
               "tokenizer so it doesn't waste time on rows that get dropped anyway."])],
          [P("It's a blunt heuristic, not a spam detector. Real short tweets get dropped and smarter spam still gets in. "
             "Building a proper spam classifier was out of scope at that point. And there was a new problem hiding in the "
             "leftovers (Mark 6).")],
          [P(f"<b>On all 1.75M tweets:</b> 159,678 had 5 words or fewer and 282,818 were repeats, leaving 1,305,322. "
             f"<b>Fake-perfect users</b> (every tweet identical) went from {u4['all_identical_users']:,} to "
             f"{u5['all_identical_users']}. That's by design, since the dedupe collapses them to one tweet. "
             f"But now {u5['mostly_non_english_users']:,} of the {u5['users_2plus']:,} remaining users "
             f"({pct(u5['mostly_non_english_share'])}) post mostly in other languages. <b>Top 100:</b> copy-paste "
             f"accounts {pct(t4['top_copy_paste'])} → {pct(t5['top_copy_paste'])}, but "
             f"{pct(t5['top_non_english'])} of the new top 100 post mostly in other languages. Near-duplicate spam "
             "(same text, tiny changes) also still gets through.")],
          "Find out why some of the worst predictions were coming out near-random.")

s += mark("Mark 6: English only", "Dec 6–9 · commits 4b58e61 → 8a5e937 · 4 commits",
          [P("Spot-checking the worst predictions showed they were Spanish and Italian election tweets (4b58e61). BERTweet "
             "learned from English tweets, so on these it's just guessing. Added a <font name='DVM'>langdetect</font> filter "
             "(6bd0bda). It crashed on the full CSVs because it throws an error on very short text, so I caught that and "
             "treated those as non-English. I also moved it to run <i>after</i> the cheap filters, since it's by far the "
             "slowest step (5faf6d8).")],
          [P("It throws away a lot of data. And langdetect isn't perfect: in a spot-check of 40 flagged tweets (done for "
             "this report), 36 really "
             "were foreign (Spanish, Italian, German, French, Portuguese, Arabic…) but 4 were English, mostly "
             "hashtag-heavy or ALL CAPS (“SHOUT OUT TO ALL MY FELLOW SLEEP DEPRIVED… ORGANIZERS!”, "
             "“Dude. You lost.”). So about 1 in 10 removals is collateral damage.")],
          [P(f"<b>On all 1.75M tweets:</b> 350,610 of the remaining 1,305,322 tweets (27%) were flagged non-English, leaving "
             f"954,712. Mostly-non-English users went from {u5['mostly_non_english_users']:,} (Mark 5) to "
             f"{u6['mostly_non_english_users']}, leaving {u6['users_2plus']:,} users with 2+ tweets to rank. "
             f"<b>Top 100:</b> mostly-non-English {pct(t5['top_non_english'])} → {pct(t6['top_non_english'])}, "
             f"copy-paste {pct(t6['top_copy_paste'])}. But the top 100 still had a <b>median of "
             f"{t6['top_count_median']:.0f} tweets each</b>: ranking by lowest spread rewards tiny accounts, because two "
             "similar tweets look perfectly consistent. Your commit notes say one full run took about 5 hours on the 3070.")],
          "Tidy it into proper stages, give the model a real spam class instead of relying on word counts, and replace "
          "the lowest-spread ranking with a score that accounts for how many tweets someone posted.")
s += [fig("fig7_users.png", "Figure 6. Same 1.75M tweets; only the cleaning changes. Left: who makes up the top 100 "
          "“most polarized” users when you rank by lowest spread (every tweet scored by the Mark 4 model). "
          "Middle and right: counts across all users. Accounts that post one tweet on repeat get a fake “perfect” "
          "zero spread; non-English tweets are ones BERTweet can't read.")]

# 6 final
fp = fin["test_heldout"]["pol_acc"]
s += [CondPageBreak(3.2 * inch), P("6. The final pipeline and results", "h1"),
      P("Dec 10–15, 9 commits, plus later changes in the desktop copy. This round was less about new ideas and more "
        "about making the thing solid:", "body"),
      *B(["<b>Proper stages.</b> Split into stage files with all settings in one config.py, and cached the cleaned data so "
          "reruns skip the slow language detection.",
          "<b>A real spam class.</b> The model got a third output, Spam, trained on the ~90 spam tweets I'd labeled back "
          "in Mark 4. Class weights stop it from ignoring such a small class. Stance gets renormalized over just the two "
          "political classes, so a pro-Trump tweet with some spam-ish phrasing still reads as strongly pro-Trump (1a7975d).",
          "<b>Way more training data.</b> On top of the hand labels (counted 5×), it trains on ~990k tweets labeled by "
          "which hashtag file they came from, with label smoothing because those labels are noisy.",
          "<b>A better polarization score.</b> Sorting by 1/spread let a user with 3 identical tweets beat one with 50 "
          "consistent ones. New score = 0.6 × (low spread) + 0.2 × (model confidence) + 0.2 × (share of "
          "tweets on their majority side), times a log tweet-count weight (c46e6a5).",
          "<b>Population view.</b> Fit a 2-cluster Gaussian mixture to all users' average stance and use the overlap "
          "between the clusters as the headline number. Less overlap means a more split population.",
          "<b>Leakage fix</b> (desktop version). Hand labels used to be copied 5× <i>before</i> the train/validation "
          "split, so copies of the same tweet landed on both sides and validation scores looked better than reality. Now "
          "the copies are made after the split, training side only."]),
      box([P(f"<b>Results.</b> Stance accuracy on the 155 held-out tweets: <b>{pct(fp)}</b> (vs {pct(ful)} for Mark 4; both "
             "drop to about 7 out of 12 on brand-new tweets, see section 7). "
             "From the 2nd full run: 30,525 users kept after cleaning and spam filtering. Two clusters: pro-Biden centered "
             "at −0.42 (53% of users) and pro-Trump at +0.31 (47%), with an overlap of 0.077. Heads up: these user "
             "numbers didn't survive a later check. See section 6b.")], border=BLUE),
      Spacer(1, 8),
      fig("fig6_final.png", "Figure 7. Left: every user's average stance with the two fitted clusters (from your "
          "user_polarization_scores_2nd_run.csv, refit here and it reproduces the 0.077). Right: before the tweet-count "
          "weight, busier users are slightly <i>less</i> polarized; the upward trend only appears after multiplying in "
          "log(tweet count). See section 7.")]

# 6b mark 7
s += [CondPageBreak(3.2 * inch), P("6b. Mark 7: the shortcut and the fix", "h1"),
      P("October 2026.", "small"),
      P("Building the demo is what broke this open. Typing “I love Biden” came back 80% Trump on Mark 4, and "
        "“I love trump” came back 98% Biden on the saved final model. Two different problems were hiding behind good "
        "test scores:"),
      *B(["<b>Mark 4 learned who's mentioned, not what's said.</b> In the hand labels, 80% of pro-Trump tweets mention Biden "
          "and 85% of pro-Biden tweets mention Trump. Tweets naming only Biden are pro-Trump 83% of the time, and tweets naming "
          "only Trump are pro-Biden 72% of the time. Hardly any praise anyone (positive words in 4–8%). So “mentions "
          "Biden = Trump fan” really works on election Twitter, right up until someone says something nice.",
          "<b>The old final model seems to have learned which file a tweet came from.</b> Its labels were the source file, and its spam "
          "head collapsed, so it snaps to about two answers no matter what you type.",
          "<b>The test set couldn't catch either one.</b> It came from the same pile, with the same attack-heavy mix, so it "
          "rewarded the shortcut. The scores were real. They just couldn't see this."]),
      P("The fix", "h3"),
      *B(["<b>Better labels.</b> Hashtags that pick a side (#VoteBiden, #Trump2020) instead of the source file. Checked against "
          "the hand labels, they agree 80% of the time, versus 29% for the source file.",
          "<b>Hide the answer.</b> The side tag gets deleted from 3 of every 4 training tweets, so the model has to read the words.",
          "<b>Break the shortcut on purpose.</b> Tweets are grouped by who they mention, and each group gets equal counts per side "
          "(12,000 each, 72,000 total). Now a name alone tells the model nothing.",
          "<b>Trust the hand labels more.</b> Counted ×20 instead of ×5, and they alone pick the best checkpoint.",
          "<b>Lock the tests.</b> Test tweets are filtered out of training by a normalized match, and a near-duplicate search "
          "afterward found 0 exact leaks. Dropping the few near-duplicates barely moves any score."]),
      fig("fig8_mark7.png", "Figure 9. Same three models on four tests. Held-out is your hand labels. The other three were built to "
          "be harder and don't come from the training slice. Mark 7 here is the first run, the one the demo uses."),
      box([P(f"<b>Results.</b> Held-out: {pc0(E7['heldout']['Mark 4']['acc'])} → <b>{pc0(H7.mean())}</b> (mean of {len(H7)} runs, {pc0(H7.min())} and {pc0(H7.max())}). "
             f"Tweets that support a candidate and name only that candidate: {pc0(E7['shortcut']['Mark 4']['acc'])} → <b>{pc0(E7['shortcut']['Mark 7']['acc'])}</b>. "
             f"Fresh tweets written before training: {pc0(E7['fresh']['Mark 4']['acc'])} → <b>{pc0(E7['fresh']['Mark 7']['acc'])}</b>. "
             f"Random untagged tweets: {pc0(BL7['Mark 4']['acc'])} → <b>{pc0(BL7['Mark 7']['acc'])}</b> (always saying Biden gets {pc0(BL7['always_biden'])}). "
             f"A second run got {', '.join(pc0(x) for x in ALL7['shortcut'])} on supportive, {', '.join(pc0(x) for x in ALL7['fresh'])} on fresh and "
             f"{', '.join(pc0(x) for x in ALL7['blind'])} on untagged. A third retrain got cut off by low memory, so that's two runs, not many.")], border=BLUE),
      Spacer(1, 8),
      P("What it did to the user results", "h3"),
      P(f"The old user numbers don't hold up. Among users who kept declaring a side with hashtags, the 2nd run's score matched their "
        f"side only {pc0(UC7['old 2nd run']['acc'])} of the time. Mark 7 matches {pc0(UC7['Mark 7']['acc'])}, or "
        f"{pc0(UC7['Mark 7 tag-free']['acc'])} when scoring only their tweets with no side tags, which training never saw. "
        "31% of the users in both runs switched sides. New run: 27,965 users, a pro-Biden cluster at −0.60 (68%) and a "
        "pro-Trump cluster at +0.59 (32%), overlap 0.011."),
      P("Caveats to keep", "h3"),
      *B(["<b>Don't claim the population got more polarized.</b> Overlap fell from 0.077 to 0.011, but Mark 7 is also more "
          "confident, which pushes scores outward, so the two overlaps aren't comparable. BIC still prefers 3 clusters over 2 "
          "(33,834 vs 36,759).",
          "<b>The 68/32 split is about this dataset,</b> not US voters. People tweeting these hashtags aren't a poll.",
          "<b>The new tests have limits.</b> Supportive tweets were chosen for being "
          "clear, and 43 of the 120 random tweets were skipped as neutral, foreign or unclear, so 90% means tweets with a clear side. "
          "Re-check <font name='DVM'>data/shortcut_test.csv</font> and <font name='DVM'>data/blind_test.csv</font> by hand before you lean on them.",
          "<b>The 64% vs 89% isn't a perfect match-up.</b> The old checkpoint is gone, so it can't be rerun tag-free. Its 64% had tags "
          "in the text, which should have helped it if anything.",
          "<b>Spam is working, barely.</b> The head no longer collapses, but it caught 6 of 14 spam tweets, with 6 false alarms out of 155.",
          "<b>It still misses things.</b> “Those MAGA people are crazy” is a 53/47 coin flip it loses."])]

# 7 findings
s += [P("7. Things found while rerunning everything", "h1"),
      P("Read these before you write. A couple of them change which numbers you can safely claim.")]
finds = [
    ("The saved model is the broken one.", "model/bertweet_finetuned/best_model.pt (saved Apr 20, 13:04) predicts Spam for "
     "100% of tweets (average P(spam) = 0.98). That's the class-weight collapse your own comment in stage3_training.py warns "
     "about. It's why user_polarization_scores.csv and stance_histogram.png from 15:23 that day are empty. The good "
     "_2nd_run results came from an earlier checkpoint that got overwritten. The Biden-vs-Trump half of the saved model "
     f"still works ({pct(fp)} held-out) because stance only compares those two classes. Those 2nd-run results later failed "
     "a check against users' declared sides, so quote Mark 7's instead (section 6b)."),
    ("Hashtag labels are much noisier than assumed.", "The code assumes the weak labels (“it's in the #trump file, so "
     "it's pro-Trump”) are 20–30% wrong. In your 1,000 hand labels they were wrong 71% of the time. Those are the "
     "first 500 rows of each file, probably all from the same few days (lots of Hunter Biden email tweets), so the full "
     "dataset may be less extreme, but it's worth checking and worth saying in the write-up. It also makes the hand-labeling "
     "work look even more important."),
    ("“Two camps” is a fit, not a proof.", "The overlap number is computed on weighted curves, so its maximum is "
     "about 0.5, not 1. And a model comparison score (BIC, lower = better) prefers 3 clusters over 2 (32,556 vs 33,633) "
     "and 2 over 1 (37,249). Fair to say: two clusters fit much better than one. Not fair to say: the data shows exactly two."),
    ("The activity correlation is mostly baked in.", "Tweet count vs final polarization score: r = 0.146. But the final score "
     "multiplies in log(tweet count), so part of that is automatic. On the raw score it's r = −0.05."),
    ("It's best on tweets like its training data.", "All 1,000 hand labels came from the first 500 rows of each file, "
     "which look like the same few days in mid-October 2020. The held-out test tweets come from that same slice, so the "
     "76–80% is “in-distribution”. On 12 brand-new tweets written for this report (6 obviously pro-Trump, 6 "
     "obviously pro-Biden), Mark 4 and the final model each got only 7 right. The saved final model also tends to default to "
     "Biden, because its collapsed spam head leaves only ~3% of the probability for the two political classes. This is what "
     "led to Mark 7 (section 6b). Say it in the write-up; knowing where your model breaks is a strength, not a weakness."),
    ("Mark 3 crashes on the full dataset.", "167 empty tweets + a missing str() cast. For the rerun it was fed empty strings "
     "instead of blanks; the scoring code itself was untouched."),
    ("Desktop and GitHub don't match.", "GitHub has the older model (first-token pooling, no inner dropout), so the saved "
     "best_model.pt won't even load with GitHub's model.py. GitHub is also missing the leakage fix, label smoothing, the "
     "rule-based spam flags and the cache. Going the other way, GitHub's code comments were rewritten in a casual style "
     "that the desktop copy doesn't have. A branch syncing the desktop code + the demo is ready (see section 12)."),
]
s += [table([["Finding", "Details"]] + [[f"<b>{a}</b>", b] for a, b in finds], [1.7 * inch, CONTENT_W - 1.7 * inch])]

# 8 fact check
s += [CondPageBreak(3 * inch), P("8. Fact-check of the original notes", "h1"),
      P("MIT readers can open your GitHub, so the write-up should match it. Here's where your original notes and the git "
        "history disagree:"),
      table([["Original note says", "Git / rerun shows", "Suggested wording"],
             ["“hundreds of keywords” (Mark 1)", "48 unique keywords (26 Trump, 23 Biden, one repeated)",
              "“about 50 keywords”"],
             ["Mark 2 “increased the number of keywords by 5 times”", "5fd63dd went from 48 to 70 (about 1.5×), "
              "then 22f45be reverted it. Mark 2's final version has the same 48.",
              "“tried adding ~20 more, then rolled them back”, or just drop it"],
             ["Mark 2 “provided a noticeable improvement”", f"It fixed the MAGA example, but overall accuracy stayed at "
              f"{pct(t2['accuracy'])} and it broke the clown-car example", "“fixed the case I was targeting, but the overall "
              "number didn't move” (honestly a better story)"],
             ["Mark 4 “trained it by giving it 1000 pre classified tweets”", "1,000 hand-labeled; Mark 4 dropped the ~90 "
              "spam and trained on 80% of the rest (~730)", "“about 1,000 tweets I labeled by hand”"],
             ["Frozen encoder was “random chance” (bd4d639 commit)", f"Rerun gets {pct(frz)}, loss falls to ~0.5–0.6",
              "“freezing worked a bit; fine-tuning worked better”"],
             ["Mark 5 “only tweets with over 5 words”", "Correct, plus a dedupe of repeated (user, text) pairs",
              "mention the dedupe too"],
             ["Mark 6 is “the final product”", "One more round after it: spam class, composite score, GMM, leakage fix",
              "call it “the final version” (Mark 7 is now the shortcut fix)"],
             ["“aprox. 5 hours on an RTX 3070”", "Matches the 8a5e937 commit note", "fine as is"]],
            [1.75 * inch, 2.75 * inch, CONTENT_W - 4.5 * inch])]

# 9 mark
s += [P("9. Mark ?: what I'd do next", "h1"),
      *B(["<b>Real spam detection.</b> Mark 7's spam output no longer collapses, but it only catches 6 of 14. "
          "Label a few hundred more spam tweets, or build a small separate model on account behavior (posting rate, share of "
          "repeated tweets) instead of tweet text.",
          "<b>A bigger, double-checked test set.</b> 155 political test tweets means every percentage has about ±6 points of "
          "wiggle room (95% range). Label another 1,000 and have a second person label a chunk of them to see how often two humans agree.",
          "<b>Keep the non-English tweets.</b> A multilingual tweet model (like XLM-T) instead of throwing away 27% of the data.",
          "<b>Human-checked hard tests.</b> Have a person re-label Mark 7's supportive and blind tests, then grow them.",
          "<b>Test the two-camps claim properly</b> with a model comparison (BIC) or a bimodality test, and report it either way."])]

# 10 lessons
s += [P("10. Engineering lessons (the adapting part)", "h1"),
      table([["Lesson", "Where it happened"],
             ["<b>Measure before you fix.</b> I chased the errors I could see, not the ones that mattered most.",
              "Marks 2–3 targeted wrong-side errors (4% of tweets). The real problem was 88% no-calls, and I never "
              "measured that at the time."],
             ["<b>Know when to stop patching.</b> If three rounds of fixes don't move the number, the approach is the problem.",
              "11 rule commits moved accuracy about 1 point. Switching to ML moved it about 68."],
             ["<b>Test cheap before you rebuild.</b>", "Spiked BERTweet on a few tweets before deleting the rule code (5343537)."],
             ["<b>Every fix can break something.</b> Rerun on more data after each change.",
              "“crazy good”, the clown-car regression, the off-by-one, the langdetect crash, the spam collapse."],
             ["<b>Look at the actual output, not just the score.</b>", "Reading the top-20 users found the bots (07422d9); "
              "reading the worst predictions found the foreign tweets (4b58e61)."],
             ["<b>Order and cache by cost.</b>", "Cheap filters before langdetect; cache the cleaned data so reruns skip it."],
             ["<b>Don't let test data leak into training.</b>", "Hand labels copied before the split inflated validation scores; "
              "fixed by copying after."],
             ["<b>A test from the same pile can't catch a shortcut.</b>", "76–80% looked great. Typing “I love Biden” "
              "found what the test couldn't, and a test built to break the shortcut took Mark 4 down to 13%."],
             ["<b>Be suspicious of a big jump.</b>", "Mark 7's scores looked too good, so it got a leak audit, a blind test and "
              "a second retrain before any of them got reported."]],
            [2.6 * inch, CONTENT_W - 2.6 * inch])]

# 11 MIT
s += [P("11. Using this for the MIT Maker Portfolio", "h1"),
      P("MIT says the portfolio is about what you made, how you made it, and what you learned, and that process "
        "(including what failed) counts. This project is strong on exactly that, so lead with the iterations, not just "
        "the final number. Check the current instructions on the MIT Admissions site for exact word and image limits, "
        "because those change."),
      P("Good images to use", "h3"),
      *B(["Figure 1 (accuracy by Mark) as the headline image. One glance shows the whole arc.",
          "Figure 3 (pipeline evolution) to show the design changes.",
          "The case-study table from Mark 3, since it's the clearest example of a fix that broke something.",
          "Figure 6 (bots and non-English users per Mark) for the spam/language story.",
          "Figure 9 (Mark 4 vs Mark 7 on the hard tests) for the shortcut story.",
          "A screenshot of your GitHub commit history. It's honest evidence of the process."]),
      P("Be clear about who did what", "h3"),
      *B(["Say which parts used AI tools. Your notes say the keyword lists were partly AI-generated; say so.",
          "Credit the pretrained model: BERTweet (Nguyen, Vu &amp; Nguyen, 2020; vinai/bertweet-base on Hugging Face).",
          "Credit the dataset. The two CSVs look like the public Kaggle “US Election 2020 Tweets” set "
          "(hashtag_donaldtrump / hashtag_joebiden). Confirm where you got them and cite that.",
          "The parts that are clearly yours: the 1,000 hand labels, every rule version, the model head and training loop, "
          "the cleaning, the scoring formula, the original debugging, and catching that the demo's answers were wrong."]),
      P("A one-line pitch you could adapt", "h3"),
      box([P("“I built a pipeline to measure political polarization in 1.75 million election tweets, rebuilt it seven "
             "times as each version broke in a new way, and took held-out accuracy from 8% with keyword rules to 87%. The last "
             "fix came from typing “I love Biden” into my demo and watching a model with a good test score get it wrong.”")], border=AQUA)]

# 12 demo codebase
s += [CondPageBreak(4 * inch), P("12. Demo and codebase (what MIT asks for)", "h1"),
      P("For code projects MIT wants the codebase (ideally version-controlled) <i>and</i> a working demo (video, live, or "
        "executable). You now have all three kinds:"),
      *B(["<b>Video:</b> <font name='DVM'>Polarization Pipeline Demo.mp4</font> on your desktop (captioned). It runs "
          "the write-up's example tweets through every version, scores a real user and a copy-paste bot, and grades the "
          "models on random held-out tweets. Upload it to YouTube (unlisted) or wherever the portfolio form takes a link.",
          "<b>Executable / live:</b> <font name='DVM'>python demo/app.py</font> opens the web demo; "
          "<font name='DVM'>--cli \"tweet\"</font> prints every version's call in the terminal; "
          "<font name='DVM'>--share</font> makes a temporary public link (about a week) if a reviewer wants to try it live.",
          "<b>Codebase:</b> the GitHub repo with all 46 commits. A branch, <font name='DVM'>desktop-sync-and-demo</font>, adds "
          "the desktop code (the version the saved model needs), the demo, Mark 7 and the evaluation scripts. It's committed "
          "locally, with its real dates, but not pushed yet."]),
      P("Marks 1–3 in the demo are the original code copied unchanged from their commits, so the demo also proves the "
        "history is real: anyone can watch Mark 2 break the clown-car tweet. The demo is honest about the limits too. The "
        "last tab's draws are truly random, so the tally moves around on a short run, and Mark 7's misses stay visible."),
      fig("../demo_mark7/shot_tab1_3.png", "Figure 10. The demo's first tab on “I love Biden”: the rules give up, Mark 4 and "
          "the old final model say Trump, Mark 7 gets it.", w=CONTENT_W * 0.9)]

# appendix
rows = [l.strip().split("|", 2) for l in open(os.path.join(W, "commits.txt"), encoding="utf8") if l.strip()]
starts = {"da08515": "1", "112781e": "2", "9f440ef": "3", "5343537": "4", "1503a65": "5", "4b58e61": "6",
          "f5a0e74": "Final", "8ce8559": "Publish"}
cur, crow = None, [["Mark", "Commit", "Date", "Message"]]
for h, d, m in rows:
    cur = starts.get(h, cur)
    crow.append([cur, f"<font name='DVM'>{h}</font>", d[:10], m.replace("&", "&amp;").replace("<", "&lt;")])
s += [PageBreak(), P("Appendix A. Every commit", "h1"),
      P(f"All {len(rows)} commits on GitHub in order, grouped by Mark. Handy for pointing at specific changes in the write-up.",
        "small"),
      table(crow, [0.55 * inch, 0.75 * inch, 0.95 * inch, CONTENT_W - 2.25 * inch])]

s += [P("Appendix B. Test details", "h1"),
      *B(["<b>Hand labels:</b> data/manual_labels.csv, 1,000 rows (874 unique texts): 438 pro-Trump, 365 pro-Biden, 71 spam "
          "after removing duplicates. One text had two conflicting labels.",
          "<b>Held-out test set:</b> the 169 hand-labeled tweets in the final model's validation split (rebuilt with "
          "train_test_split, 80/20, stratified, random_state=42, over the same ~990k combined rows). 82 pro-Trump, 73 "
          "pro-Biden, 14 spam.",
          "<b>Accuracy:</b> computed on the 155 political tweets. A 0 / no call counts as wrong. The final model's "
          "side is P(Trump) vs P(Biden), ignoring its broken spam output.",
          "<b>Mark 4 reruns:</b> e554bad code, 5 seeds: " + ", ".join(pct(x, 1) for x in m4) +
          ". Frozen (bd4d639), 3 seeds: " + ", ".join(pct(x, 1) for x in m4f) + ".",
          "<b>User checks (Figure 6):</b> run on every tweet with Mark 5/6's own clean.py (commit 8a5e937). Only users "
          "with 2+ tweets count, since a spread needs two. “Fake-perfect” = all of a user's tweets are the same "
          "text after stripping links. “Mostly non-English” = langdetect flags over half the user's tweets. "
          "Top-100 ranking: one Mark 4 model (seed 1) scores all 1.4M unique cleaned tweets, users with 2+ tweets are "
          "sorted by lowest spread as Mark 4 did; “copy-paste account” = at least half the user's tweets repeated.",
          "<b>Fresh-tweet probe:</b> 12 sentences written for this report (portfolio_eval/probe_side.py); a model counts as right if it picks the obvious side.",
          "<b>Scripts:</b> everything used to produce these numbers is saved in political_polarization/portfolio_eval/ on "
          "your desktop, so it can be rerun."])]

doc.build(s)
print("wrote", OUT)
