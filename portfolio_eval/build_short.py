# build_short py the
import json, os, glob
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether)
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
    "mk": ParagraphStyle("mk", fontName="DV-B", fontSize=9.2, leading=11.5, textColor=INK, spaceBefore=5, spaceAfter=1),
    "res": ParagraphStyle("r", fontName="DV", fontSize=8.6, leading=11.6, textColor=BLUE, leftIndent=10, spaceAfter=3),
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


# mark 7 numbers
_R = r"C:\Users\Asus Gaming\Desktop\political_polarization\portfolio_eval"
_j = lambda n: json.load(open(os.path.join(_R, n)))
E7, BL7, UC7, SD7 = _j("mark7_eval.json"), _j("blind_eval.json"), _j("user_check.json"), _j("mark7_seeds.json")
H7 = np.array([E7["heldout"]["Mark 7"]["acc"]] + [v["heldout"] for v in SD7.values()])
ALL7 = {k: [v[k] for v in SD7.values()] for k in ("heldout", "shortcut", "fresh", "blind")}
pc0 = lambda x: f"{x*100:.0f}%"


def on_page(c, doc):
    c.saveState(); c.setFillColor(PAPER); c.rect(0, 0, letter[0], letter[1], stroke=0, fill=1)
    c.setFont("DV", 7); c.setFillColor(MUTED)
    c.drawRightString(letter[0] - 0.65 * inch, 0.4 * inch, f"{doc.page}")
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
        "· Mark 7 added Oct 2026 · Python, PyTorch, BERTweet · live demo in demo/", "meta"),
      P("<b>The goal:</b> measure how politically polarized Twitter users were during the 2020 US election. I wanted to "
        "know whether each <i>person</i> stayed on one side, and whether the population split into two camps, not just "
        "label single tweets. The data was 1.75 million tweets from #trump and #joebiden. I rebuilt the pipeline seven times. "
        "Each version fixed what broke in the one before, and I re-measured every version on the same 155 hand-labeled "
        "test tweets.")]

img = os.path.join(FIG, "fig2_accuracy.png"); iw, ih = PI.open(img).size; w = CW * 0.62
s += [Table([[Image(img, width=w, height=w * ih / iw)]], colWidths=[CW],
            style=[("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
      P("Same 155 held-out tweets for every version. A “no call” counts as wrong. Mark 4 is the mean of 5 retrains, Mark 7 of 2.", "cap")]

u4 = U["Mark 4"]; T4, T5, T6 = TOP["Mark 4"], TOP["Mark 5"], TOP["Mark 6"]


def mark(name, broke, fix, better):
    # one mark what
    return [KeepTogether([P(name, "mk"), P(f"<b>Problem:</b> {broke} <b>Fix:</b> {fix}"), P(f"→ {better}", "res")])]


s += [P("How each version got better", "h")]
s += mark("Mark 1 · keyword matching",
          "I needed a first way to tell sides apart, and pro-side tweets seemed to share hashtags.",
          "score each tweet by about 50 keywords like #MAGA and #VoteBiden.",
          f"{pc(t['1ba92a1']['accuracy'])} correct. When it made a call it was right {pc(t['1ba92a1']['precision_called'])} "
          f"of the time, but it couldn't decide on {pc(1-t['1ba92a1']['coverage'])} of tweets.")
s += mark("Mark 2 · negative adjectives",
          "“Those MAGA people are crazy” scored pro-Trump.",
          "flip the score when a negative adjective shows up.",
          f"That tweet is fixed, but it broke “…the trump clown car. VoteBiden”, which Mark 1 had right. Still "
          f"{pc(t['22f45be']['accuracy'])}: only 3% of tweets even have a keyword plus an adjective.")
s += mark("Mark 3 · adjectives weighted by distance",
          "tweets with adjectives aimed at different people.",
          "pair each keyword with its nearest adjective and weight by how far apart they are.",
          f"Clown-car tweet right again, {pc(t['a2f78b5']['accuracy'])} overall. Three versions of rules moved accuracy "
          "about one point, and sarcasm was hopeless. So I stopped patching and changed the approach.")
s += mark("Mark 4 · a real language model",
          "rules see words, not meaning.",
          "fine-tune BERTweet (a model pretrained on tweets) on about 1,000 tweets I labeled by hand. Labeling by hashtag "
          "would have been wrong 71% of the time, so hand labels were worth the hours.",
          f"{pc(t['a2f78b5']['accuracy'])} → <b>{pc(m4.mean())}</b> (±{m4.std()*100:.0f} over 5 retrains), and it makes a call "
          f"on every tweet. But scoring users exposed a new bug: {u4['all_identical_users']:,} accounts posted one tweet on "
          f"repeat, zero spread looked “perfectly polarized”, and they filled {pc(T4['top_copy_paste'])} of my top 100.")
s += mark("Mark 5 · cleaning",
          "spam bots topped the ranking.",
          "keep tweets with more than 5 words and drop repeated (user, text) pairs.",
          f"Repeat-bot accounts {u4['all_identical_users']:,} → <b>0</b>; copy-paste share of the top 100 "
          f"{pc(T4['top_copy_paste'])} → <b>{pc(T5['top_copy_paste'])}</b>. Next bug: {pc(T5['top_non_english'])} of the new "
          "top 100 mostly tweeted in other languages.")
s += mark("Mark 6 · English only",
          "the worst predictions were Spanish and Italian tweets.",
          "a language filter, run after the cheap filters so it only checks what's left.",
          f"Non-English in the top 100 {pc(T5['top_non_english'])} → <b>{pc(T6['top_non_english'])}</b> (36 of 40 flagged tweets "
          f"I checked really were foreign). Next bug: the top 100 had a median of {T6['top_count_median']:.0f} tweets each.")
s += mark("Final version · weighting, spam class, population model",
          "the ranking rewarded tiny accounts, and I had no view of the whole population.",
          "weight each user's score by how much they tweet, add a spam class, train on weak labels from 990k tweets plus my "
          "hand labels, and fit a two-cluster model to all users.",
          f"<b>{pc(fin)}</b> on held-out tweets, and 30,525 users split into two camps. It looked done. Then I typed "
          "“I love Biden” into my demo and it said Trump.")
s += mark("Mark 7 · breaking the shortcut",
          "tweets naming only Biden are pro-Trump 83% of the time, so the models learned “mentions Biden = Trump fan”. "
          "The test tweets had the same mix, so the score never showed it. The saved final model had also collapsed.",
          "label by hashtags that pick a side (they match my hand labels 80% of the time, versus 29% for the old labels), delete "
          "the tag so the model has to read the words, balance training so a name alone tells it nothing, and lock the tests "
          "out of training.",
          f"Held-out {pc(E7['heldout']['Mark 4']['acc'])} → <b>{pc(H7.mean())}</b> ({pc0(H7.min())} and {pc0(H7.max())} on 2 runs). Supportive tweets "
          f"{pc(E7['shortcut']['Mark 4']['acc'])} → <b>{pc(E7['shortcut']['Mark 7']['acc'])}</b>. Fresh tweets "
          f"{pc(E7['fresh']['Mark 4']['acc'])} → <b>{pc(E7['fresh']['Mark 7']['acc'])}</b>. Random untagged tweets "
          f"{pc(BL7['Mark 4']['acc'])} → <b>{pc(BL7['Mark 7']['acc'])}</b>. User scores matching the side people declared with hashtags: "
          f"{pc(UC7['old 2nd run']['acc'])} → <b>{pc(UC7['Mark 7 tag-free']['acc'])}</b>. Those extra tests were labeled by AI, not by me, and "
          "only cover tweets with a clear side.")

s += [P("How I worked", "h"),
      P("Same loop every time: <b>run it, read the actual output, make the smallest fix, retest on more data.</b> "
        "None of the bugs showed up in an accuracy number. I found them by reading tweets, reading the top-20 users, and "
        "reading the worst predictions. Retesting is how I caught Mark 2 breaking a tweet Mark 1 had right. The hardest "
        "call was knowing when to stop patching: once three fixes barely moved the number, the approach was the problem, "
        "and switching it was worth about 67 points. The biggest lesson came last: a test from the same pile as the training "
        "data can't catch a shortcut."),
      P("What's next", "h"),
      P("A better spam detector (Mark 7 catches only 6 of 14), my own labels on the new tests, and a multilingual model so 27% "
        "of the data isn't thrown away."),
      P("Evidence", "h"),
      P("Code and all 46 commits on GitHub, a 100-second demo video, and a live demo "
        "(<font name='DVM'>python demo/app.py</font>) that runs one tweet through every version, using the original "
        "Mark 1–3 code checked out from git. Every number here came from rerunning that version.")]

doc.build(s)
print("wrote", OUT)
