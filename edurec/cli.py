"""Command line demo of the final recommender.

    python -m edurec.cli learner 111316           recommendations for a learner in the dataset
    python -m edurec.cli videos 510 511 512       recommendations after watching these videos
    python -m edurec.cli new                      what a brand-new learner sees
    python -m edurec.cli video 510                details of one video
"""

import argparse
import sys
import textwrap

import numpy as np

from .service import RecommendationService, UnknownVideo


def show(service, history, k, title):
    result = service.recommend(history, k)
    print(f"\n{title}")
    print("=" * len(title))
    if len(history):
        print(f"Videos opened so far: {len(history)} (history group {result['history_group']})")
        print("Most recent:")
        for video in service.recent_history(history, 3):
            print(f"  - {video['name']} [{video['software'] or 'no software tag'}]")
        weights = ", ".join(f"{name} {value:.1f}" for name, value in result["weights"].items())
        print(f"Signal weights used: {weights}")
    else:
        print("No history yet, so these are the videos trending over the last month.")
    print(f"\nTop {k} recommendations ({result['milliseconds']} ms):")
    for item in result["recommendations"]:
        print(f"{item['rank']:>3}. {item['name']}  [{item['software'] or '-'}, {item['type']}]")
        print(textwrap.indent(f"why: {item['reason']}", "      "))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pathway-aware hybrid recommender for MARS")
    commands = parser.add_subparsers(dest="command", required=True)
    learner = commands.add_parser("learner", help="recommendations for a learner id from the dataset")
    learner.add_argument("learner_id", type=int)
    videos = commands.add_parser("videos", help="recommendations for a list of watched video ids")
    videos.add_argument("video_ids", type=int, nargs="+")
    commands.add_parser("new", help="recommendations for a brand-new learner")
    video = commands.add_parser("video", help="show one video")
    video.add_argument("video_id", type=int)
    for sub in (learner, videos, commands.choices["new"]):
        sub.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)

    service = RecommendationService()
    try:
        if args.command == "learner":
            history = service.history_of(args.learner_id)
            if history is None:
                print(f"Learner {args.learner_id} is not in the dataset.")
                return 1
            show(service, history, args.k, f"Learner {args.learner_id}")
        elif args.command == "videos":
            show(service, service.indices_for(args.video_ids), args.k, "Custom history")
        elif args.command == "new":
            show(service, np.array([], dtype=int), args.k, "Brand-new learner")
        elif args.command == "video":
            index = service.indices_for([args.video_id])[0]
            row = service.catalogue.items.loc[index]
            print(f"{row['name']} (id {row['item_id']})")
            print(f"Type: {row['type']}  Software: {row['software'] or '-'}  Difficulty: {row['difficulty'] or '-'}")
            print(textwrap.fill(row["description_clean"][:400] + "...", 100))
    except UnknownVideo as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
