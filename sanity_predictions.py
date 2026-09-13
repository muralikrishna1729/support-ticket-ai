"""Sanity checks: run the tricky tickets that motivated the v3 recipe
(negation pair, paraphrase pair, typos) through the real PredictPipeline."""

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline.predict_pipeline import PredictPipeline

TICKETS = [
    # paraphrase pair
    "screen black nothing showing",
    "monitor wont turn on",
    # negation pair
    "i cannot access my account",
    "i can access my account now",
    # typos
    "recieved damaged prodct plz refund",
    # extra domain probes
    "vpn not connecting from office network",
    "payment charged twice for same order",
    "server down since morning everyone unable to login",
]

def main():
    pipe = PredictPipeline()
    for t in TICKETS:
        r = pipe.predict(t)
        print(f"\nTICKET: {t!r}")
        print(f"  category   : {r['category']}")
        print(f"  issue_type : {r['issue_type']} | confidence={r['confidence']} "
              f"| needs_review={r['needs_review']}")
        print(f"  response   : {r['auto_response']}")
        print(f"  clean      : {r['clean_text']!r}")

if __name__ == "__main__":
    main()
