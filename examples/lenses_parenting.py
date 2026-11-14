"""A worked, domain-specific Lenses config — an intergenerational *parenting*
memoir (present: parenting your own kids; past: how you were parented).

This is the kind of thing that used to be hardcoded at the top of memoir_recon.py
and spine_proposer.py. It lives here as an *example of customization*, not in the
library: swap the seeds and aspects for whatever your memoir is about.

    from examples.lenses_parenting import LENSES
    report = recon_report(entries, embed, lenses=LENSES)
"""
from datetime import date
from monograph.types import Lenses, StyleProfile

# Age context for the write stage (name -> birthdate). Pass as people= to
# MemoirEngine so a child's age at each entry's date is always right.
PEOPLE = {
    "Cash":    date(2007, 1, 10),
    "Camille": date(2008, 1, 31),
    "Ruby":    date(2010, 4, 15),
}

# Sample the author's voice from older entries, where it's truest.
VOICE_WINDOW = ("2018-01-01", "2021-12-31")

# The voice characterization that used to be hardcoded in memoir.py — now a
# StyleProfile the engine drops into its prompt.
STYLE = StyleProfile(
    name="author",
    notes=("Her voice is dry and sarcastic, with dark humor. She makes light of hard "
           "things and deflects pain with wit — MATCH that voice; the sarcasm and humor "
           "are essential to sounding like her."),
)


LENSES = Lenses(
    aspects={
        "pride":       "pride in my child, so proud of them, celebrating who they are",
        "cycle":       "the parent I want to be, breaking the cycle, not repeating how I "
                       "was raised, doing it differently than my own parents did",
        "repair":      "repairing things with my child after I got it wrong, apologizing "
                       "to my kid, reconnecting after a rupture with them",
        "temper":      "losing my temper, yelling at my kids, snapping, and the guilt after",
        "overwhelm":   "exhausted, depleted, no patience left, burned out, stretched too thin",
        "worry":       "worried and afraid for my child, scared something is wrong",
        "affection":   "a tender, close moment with my child, warmth, holding them",
        "discipline":  "setting a limit, a consequence, correcting behavior",
        "letting_go":  "my child growing up, becoming independent, learning to let them go",
        "faith":       "my faith, God, church, prayer, what I believe and how it steadies me",
    },
    emo_pos=["joy and happiness", "love and tenderness", "pride", "gratitude", "calm and peace"],
    emo_neg=["anger and frustration", "fear, anxiety and worry", "sadness and grief",
             "guilt and shame", "exhaustion and being overwhelmed"],
    strands={
        "present": ["parenting my children", "being a mom to my kids", "raising my children",
                    "disciplining my child", "comforting my child when upset",
                    "bedtime with the kids", "losing my patience with my kids",
                    "apologizing to my child", "being proud of my kid",
                    "worrying about my child", "being a single parent"],
        "past": ["how I was parented", "the way I was raised", "when I was a child",
                 "growing up", "my mother", "my father", "my parents", "my childhood",
                 "how my parents treated me", "my own upbringing"],
    },
    name_stop=Lenses.default().name_stop | {"Mom", "Dad", "Happy", "Christmas",
                                            "Thanksgiving", "Easter", "New", "Year"},
)
