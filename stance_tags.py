# hashtags that pick
import re
from stage0_cleaning import clean_text

PRO_BIDEN = set("""bidenharris2020 biden2020 bidenharris bidenharis2020 joebidenkamalaharris2020 joebiden2020
bidenharristosaveamerica bidenharris2020tosaveamerica votebidenharristosaveamerica votebidenharris2020 votebidenharris
votebiden voteblue votebluetosaveamerica votebluetoendthisnightmare bidenharrislandslide2020 bluewave bluewave2020
votehimout byebyetrump dumptrump trumpout removetrump votetrumpout votetrumpout2020 teamjoe ridinwithbiden
trumpislosing trumpmeltdown trumpisalaughingstock trumpcrimefamily trumpisanationaldisgrace trumpispathetic
trumpvirus trumpcollapse trumptantrum""".split())
PRO_TRUMP = set("""trump2020 maga maga2020 trump2020landslide kag kag2020 kag2020landslidevictory trump2020tosaveamerica
trump2020landslidevictory 4moreyears fourmoreyears fourmoreyearsfortrump trumppence2020 votetrump votetrump2020
votetrump2020tosaveamerica votetrumptosaveamerica votered voteredtosaveamerica voteredtosaveamerica2020 redwave
redwave2020 trumptrain trumptrain2020 keepamericagreat americafirst walkaway blexit sleepyjoe sleepyjoebiden creepyjoe
bidencrimefamiily bidencrimefamily bidencrimefamiiy bidencrimesyndicate hunterbidenlaptop laptopfromhell
stopthesteal hidinbiden""".split())


TOPIC_TAGS = {"trump", "biden", "joebiden", "donaldtrump"}
BIDEN_WORDS = re.compile(r"biden|joe\b|kamala|harris", re.I)
TRUMP_WORDS = re.compile(r"trump|donald|pence", re.I)


def model_text(text):
    # one format for
    text = re.sub(r"@USER", "", clean_text(str(text))).replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def text_key(text):
    # same tweet same
    return re.sub(r"[^a-z0-9]", "", model_text(text).lower())


def mention_cell(text):
    b, t = bool(BIDEN_WORDS.search(text)), bool(TRUMP_WORDS.search(text))
    return "both" if b and t else "biden" if b else "trump" if t else "neither"


def split_tags(text):
    # pull out what
    tags = {x.lower() for x in re.findall(r"#(\w+)", str(text))}
    b, t = tags & PRO_BIDEN, tags & PRO_TRUMP
    if bool(b) == bool(t):
        return None, str(text)  # no side or
    gone = b | t
    left = re.sub(r"#(\w+)", lambda m: "" if m.group(1).lower() in gone else m.group(0), str(text))
    return (0 if b else 1), left
