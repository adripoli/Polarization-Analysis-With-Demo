# build_short py the
import json, os, glob
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib
from PIL import Image as PI

W = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(W, "..", "figs")
OUT = os.environ.get("OUT") or os.path.join(os.path.expanduser("~"), "Desktop",
                                            "Polarization Pipeline MIT Maker Portfolio (Short).pdf")
T = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
for n, f in [("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
             ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")]:
    pdfmetrics.registerFont(TTFont(n, os.path.join(T, f)))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, INK2, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#8a8984")
BLUE, ORANGE, AQUA = colors.HexColor("#2a78d6"), colors.HexColor("#eb6834"), colors.HexColor("#1baf7a")
PAPER, RULE, TINT = colors.HexColor("#fcfcfb"), colors.HexColor("#e4e3df"), colors.HexColor("#f3f2ee")
S = {
    "title": ParagraphStyle("t", fontName="DV-B", fontSize=19, leading=22, textColor=INK, spaceAfter=2),
    "meta": ParagraphStyle("m", fontName="DV", fontSize=7.8, leading=10, textColor=INK2, spaceAfter=6),
    "h": ParagraphStyle("h", fontName="DV-B", fontSize=10.5, leading=13, textColor=INK, spaceBefore=6, spaceAfter=3),
    "body": ParagraphStyle("b", fontName="DV", fontSize=8.6, leading=11.8, textColor=INK, spaceAfter=4),
    "bul": ParagraphStyle("bu", fontName="DV", fontSize=8.6, leading=11.6, textColor=INK, leftIndent=10,
                          bulletIndent=1, spaceAfter=2.5),
    "cell": ParagraphStyle("c", fontName="DV", fontSize=7.5, leading=9.6, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="DV-B", fontSize=7.5, leading=9.6, textColor=INK),
    "num": ParagraphStyle("n", fontName="DV-B", fontSize=8.6, leading=10.4, textColor=BLUE),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=7.4, leading=9.6, textColor=INK2, spaceAfter=4),
}
CW = letter[0] - 1.3 * inch
P = lambda t, st="body": Paragraph(t, S[st])
B = lambda items: [Paragraph(i, S["bul"], bulletText="•") for i in items]

# measured numbers same
kw = {c: json.load(open(os.path.join(W, f"kw_{c}.json"))) for c in ["1ba92a1", "22f45be", "a2f78b5"]}
m4 = np.array([json.load(open(f))["pol_acc"] for f in glob.glob(os.path.join(W, "m4_full_*.json"))])
m4f = np.array([json.load(open(f))["pol_acc"] for f in glob.glob(os.path.join(W, "m4_frozen_*.json"))])
fin = json.load(open(os.path.join(W, "final_eval.json")))["test_heldout"]["pol_acc"]
U = json.load(open(os.path.join(W, "users_light.json")))
TOP = json.load(open(os.path.join(W, "users_eval.json")))["results"] if os.path.exists(os.path.join(W, "users_eval.json")) else None
pc = lambda x: f"{x*100:.0f}%"
t = {c: kw[c]["test_heldout"] for c in kw}


def on_page(c, doc):
    c.saveState(); c.setFillColor(PAPER); c.rect(0, 0, letter[0], letter[1], stroke=0, fill=1)
    c.setFont("DV", 7); c.setFillColor(MUTED)
    c.drawRightString(letter[0] - 0.65 * inch, 0.4 * inch, f"{doc.page} / 2")
    c.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=0.65 * inch, rightMargin=0.65 * inch,
                      topMargin=0.5 * inch, bottomMargin=0.55 * inch,
                      title="Polarization Pipeline MIT Maker Portfolio (Short)", author="Adrian Polishchuk Carrillo")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                                                         id="f", leftPadding=0, rightPadding=0, topPadding=0,
                                                         bottomPadding=0)], onPage=on_page)])
s = []
s += [P("Polarization Pipeline", "title"),
      P("Adrian Polishchuk Carrillo · github.com/adripoli/Polarization-on-Social-Media · 46 commits, Oct–Dec 2025 "
        "· Python, PyTorch, BERTweet · live demo in demo/", "meta"),
      P("<b>The goal:</b> measure how politically polarized Twitter users were during the 2020 US election. I wanted to "
        "know whether each <i>person</i> stayed on one side, and whether the population split into two camps, not just "
        "label single tweets. The data was 1.75 million tweets from #trump and #joebiden. I rebuilt the pipeline six times. "
        "Each version fixed what broke in the one before, and I re-measured every version on the same 155 hand-labeled "
        "test tweets that the final model never trained on.")]

img = os.path.join(FIG, "fig2_accuracy.png"); iw, ih = PI.open(img).size; w = CW * 0.86
s += [Table([[Image(img, width=w, height=w * ih / iw)]], colWidths=[CW],
            style=[("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
      P("Every version graded on the same held-out tweets. A “no call” (the program couldn't decide) counts as "
        "wrong. Mark 4 is the mean of 5 retrains.", "cap")]

u4, u5, u6 = U["Mark 4"], U["Mark 5"], U["Mark 6"]
top4 = f" In the top 100 “most polarized” users, {pc(TOP['Mark 4']['top_copy_paste'])} were copy-paste accounts." if TOP else ""
top5 = f" Copy-paste accounts in the top 100: {pc(TOP['Mark 4']['top_copy_paste'])} → {pc(TOP['Mark 5']['top_copy_paste'])}." if TOP else ""
rows = [
    ["", "What I changed", "Why (what broke before)", "Measured result"],
    ["Mark 1", "Score tweets by ~50 hashtag keywords (#MAGA, #VoteBiden…)", "Starting point: pro-side tweets seemed to share hashtags",
     f"<b>{pc(t['1ba92a1']['accuracy'])}</b> correct. Right {pc(t['1ba92a1']['precision_called'])} of the time when it called, "
     f"but no call on {pc(1-t['1ba92a1']['coverage'])} of tweets"],
    ["Mark 2", "Flip the score when a negative adjective appears", "“Those MAGA people are crazy” scored pro-Trump",
     f"<b>{pc(t['22f45be']['accuracy'])}</b>. Fixed that case but broke “…trump clown car. VoteBiden”. Only 3% of "
     "tweets have a keyword + adjective"],
    ["Mark 3", "Pair each keyword with its nearest adjective, weighted by distance", "Tweets with adjectives about different people",
     f"<b>{pc(t['a2f78b5']['accuracy'])}</b>. Clown-car fixed again, but 11 rule commits moved accuracy ~1 point. Sarcasm "
     "is impossible with rules"],
    ["Mark 4", "Fine-tune BERTweet (a transformer pretrained on tweets) on 1,000 tweets I hand-labeled",
     "Rules only see words, not meaning", f"<b>{pc(m4.mean())} ± {m4.std()*100:.0f}</b>, a call on every tweet. But "
     f"{u4['all_identical_users']:,} users ({pc(u4['all_identical_share'])}) posted one tweet on repeat and tied for "
     f"“most polarized”.{top4}"],
    ["Mark 5", "Keep tweets with >5 words; drop repeated (user, text) pairs", "Spam bots topped the polarization ranking",
     f"Repeat-bots {u4['all_identical_users']:,} → <b>0</b>.{top5} But {pc(TOP['Mark 5']['top_non_english'])} of the "
     "new top 100 post mostly in other languages"],
    ["Mark 6", "Language filter (English only)", "Worst predictions were Spanish/Italian tweets",
     f"Non-English in the top 100: {pc(TOP['Mark 5']['top_non_english'])} → <b>{pc(TOP['Mark 6']['top_non_english'])}</b>; "
     f"954,712 of 1.75M tweets left. But the top 100 had a median of just {TOP['Mark 6']['top_count_median']:.0f} tweets each"],
    ["Final", "Spam class, weak labels from 990k tweets, polarization score weighted by tweet count, 2-cluster population model",
     "Ranking rewarded tiny accounts; no population view", f"<b>{pc(fin)}</b> on held-out (7 of 12 on brand-new tweets). "
     "30,525 users scored: pro-Biden cluster at −0.42, pro-Trump at +0.31, overlap 0.077"],
]
data = [[P(c, "cellb" if (i == 0 or j == 0) else "cell") for j, c in enumerate(r)] for i, r in enumerate(rows)]
tb = Table(data, colWidths=[0.55 * inch, 2.05 * inch, 1.75 * inch, CW - 4.35 * inch], repeatRows=1)
st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
      ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
      ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE), ("LINEBELOW", (0, 0), (-1, 0), 1, INK2),
      ("LINEBELOW", (0, 3), (-1, 3), 1.2, ORANGE)]  # line where rules
for i in range(2, len(rows), 2):
    st.append(("BACKGROUND", (0, i), (-1, i), TINT))
tb.setStyle(TableStyle(st))
s += [P("How each version got better", "h"), tb]

# page 2
s += [P("How I worked", "h"),
      P("Every version followed the same loop: <b>run it, read the actual output, find what's wrong, make the smallest "
        "fix, test it on more data.</b> The bugs didn't come from accuracy numbers. They came from reading things: "
        "eyeballing tweets found the mocking-MAGA problem, reading the top-20 “most polarized” users found "
        "the bots, and spot-checking the worst predictions found the foreign-language tweets. Each fix was also tested "
        "on more data before moving on, which is how I caught that Mark 2 broke a tweet Mark 1 had right, an off-by-one "
        "in the word filter, and a crash in language detection on very short tweets."),
      P("The biggest decision was <b>when to stop patching</b>. Marks 2 and 3 targeted errors I could see, but those "
        "were rare (3% of tweets). The real problem was that the rules said nothing about 88% of tweets, and no rule "
        "touched that. After three versions moved accuracy about one point, I changed the approach instead of adding "
        "another rule. Before deleting the rule code, I ran a quick test that BERTweet's tokenizer handled hashtags and "
        "emoji. Switching to the model moved accuracy about 68 points."),
      P("The machine-learning half needed its own engineering:", "body"),
      *B(["<b>Data:</b> I hand-labeled 1,000 tweets. Labeling by hashtag would have been wrong 71% of the time, "
          "because a hashtag says who a tweet is about, not which side it's on.",
          "<b>Training:</b> freezing the pretrained model and training only a small head reached 70%; fine-tuning "
          f"everything reached {pc(m4.mean())}. I found and fixed a silent tokenizer bug that was mangling hashtags and emoji.",
          "<b>Honest testing:</b> I fixed a leak where copies of hand-labeled tweets ended up in both training and "
          "validation, which made scores look better than they were. All numbers here come from tweets the model never saw.",
          "<b>Cost:</b> cheap filters run before the slow language detector, and cleaned data is cached. A full run is "
          "about 5 hours on an RTX 3070."]),
      P("What I learned", "h"),
      *B(["<b>Measure before you fix.</b> I chased the errors I could see instead of the one that mattered most.",
          "<b>If three fixes don't move the number, the approach is the problem.</b>",
          "<b>Every fix can break something,</b> so retest on more data each time.",
          "<b>Read the output, not just the score.</b> Every major bug was found by reading examples."]),
      P("What's next (Mark ?)", "h"),
      *B(["<b>Real spam detection.</b> The spam class collapsed (only 71 examples vs. 990k political tweets). I'd label "
          "more spam, or detect it from account behavior like posting rate and repeats.",
          "<b>Labels from the whole dataset, not one slice.</b> All my hand labels came from the first 500 rows of each "
          "file, so the models score 76–80% on tweets like those but only 7 of 12 on brand-new tweets I wrote. A bigger, "
          "spread-out test set with a second labeler would also tighten the error bars (with 155 tweets, 80% could really "
          "be 74–86%).",
          "<b>A multilingual model,</b> so 27% of the data isn't thrown away.",
          "<b>Test the “two camps” claim properly.</b> A 3-cluster fit actually scores slightly better than 2."]),
      P("Evidence", "h"),
      P("Code and full commit history on GitHub. A 90-second demo video, and a live demo you can run "
        "(<font name='DVM'>python demo/app.py</font>), put one tweet through every version side by side. Marks 1–3 "
        "are the original code from git, so you can watch Mark 2 break the clown-car tweet and Mark 3 fix it, score a "
        "whole user, or grade the models on random held-out tweets. Every number above was re-measured by checking out "
        "each old commit and rerunning it.")]
img2 = os.path.join(W, "..", "demo_take4", "shot_tab1_1.png"); iw2, ih2 = PI.open(img2).size; w2 = CW * 0.72
s += [Spacer(1, 4), Table([[Image(img2, width=w2, height=w2 * ih2 / iw2)]], colWidths=[CW],
                          style=[("ALIGN", (0, 0), (-1, -1), "CENTER")]),
      P("The demo on the clown-car tweet: Mark 1 right, Mark 2 broken, Mark 3 fixed, both models right.", "cap")]

doc.build(s)
print("wrote", OUT)
