# keywords.py
#
# hand-picked + AI-suggested keyword lists to guess which candidate a tweet
# is about. counts as "pro-trump" if it has one of TRUMP_KEYWORDS,
# "pro-biden" if it has one of BIDEN_KEYWORDS.
#
# heads up: these are just hashtags/phrases people used, not sentiment
# words. a tweet can say "MAGA" and still be trashing trump - analyze_tweets.py
# tries (and kinda fails) to handle that

TRUMP_KEYWORDS = [
    "maga", "trump2020", "votetrump", "kag", "trumptrain",
    "fourmoreyears", "trumppence2020", "buildthewall", "draintheswamp",
    "americafirst", "keepamericagreat", "trump train", "trumpwon",
    "maga2020", "trumpsupporter", "backtrump", "trump won", "voteredtosaveamerica",
    "trump2020landslide", "walkaway", "lawandorder", "patriotsunite",
    "trumpisright", "presidenttrump", "45", "kag2020",
]

BIDEN_KEYWORDS = [
    "votebiden", "biden2020", "bidenharris2020", "bidenharris",
    "teamjoe", "joebiden2020", "buildbackbetter",
    "battlefortheheartofthenation", "bidenwon", "biden won",
    "bluewave2020", "biden president", "joeforpresident",
    "bidenharris", "votebluetoendthisnightmare", "votehimout",
    "trumpisnotmypresident", "dumptrump", "trumpoutnow", "byedon",
    "bidenpresident", "harrisbiden", "vote blue",
]
