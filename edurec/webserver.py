"""The web API for the final model.

    python -m edurec.webserver             serves http://127.0.0.1:8000

It has these endpoints:

    GET  /                         a small demo page for trying the recommender in a browser
    GET  /health
    GET  /learners/{learner_id}/recommendations?k=10
    GET  /learners/{learner_id}/history
    GET  /learners/heaviest        the learner with the longest history, for the demo page
    GET  /learners/random          a random learner with at least two videos, for the demo page
    POST /recommendations          body: {"history": [510, 511], "k": 10}
    GET  /videos/{video_id}
    GET  /videos?search=excel      title search, used by the demo page

I wrote it with the standard library only, so the service runs anywhere the model runs, without a web
framework.
"""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .service import RecommendationService, UnknownVideo

LEARNER_ROUTE = re.compile(r"^/learners/(\d+)/recommendations$")
HISTORY_ROUTE = re.compile(r"^/learners/(\d+)/history$")
DEMO_PAGE = Path(__file__).parent / "static" / "index.html"
VIDEO_ROUTE = re.compile(r"^/videos/(\d+)$")


class BadRequest(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def read_k(value, default=10):
    """k has to be a whole number from 1 to 50."""
    try:
        k = int(value if value is not None else default)
    except (TypeError, ValueError):
        raise BadRequest(422, "k must be a whole number") from None
    if not 1 <= k <= 50:
        raise BadRequest(422, "k must be between 1 and 50")
    return k


def make_handler(service: RecommendationService):

    class Handler(BaseHTTPRequestHandler):

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self, route):
            try:
                self.send_json(200, route())
            except BadRequest as error:
                self.send_json(error.status, {"detail": str(error)})
            except UnknownVideo as error:
                self.send_json(422 if self.command == "POST" else 404, {"detail": str(error)})

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            learner = LEARNER_ROUTE.match(url.path)
            past = HISTORY_ROUTE.match(url.path)
            video = VIDEO_ROUTE.match(url.path)
            if url.path == "/":
                self.send_page()
            elif url.path == "/videos":
                self.handle_request(lambda: {"videos": service.search(query.get("search", [""])[0],
                                                                      read_k(query.get("limit", [None])[0]))})
            elif past:
                self.handle_request(lambda: self.learner_history(int(past.group(1))))
            elif url.path in ("/learners/heaviest", "/learners/random"):
                pick = service.heaviest_learner if url.path.endswith("heaviest") else service.random_learner
                self.handle_request(lambda: self.learner_summary(pick()))
            elif url.path == "/health":
                self.handle_request(lambda: {"status": "ok", "model": service.model.name,
                                             "videos": service.catalogue.n_items, "learners": len(service.histories)})
            elif learner:
                self.handle_request(lambda: self.learner_recommendations(int(learner.group(1)), query))
            elif video:
                self.handle_request(lambda: self.video_details(int(video.group(1))))
            else:
                self.send_json(404, {"detail": "Not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/recommendations":
                self.send_json(404, {"detail": "Not found"})
                return
            self.handle_request(self.history_recommendations)

        def send_page(self):
            data = DEMO_PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def learner_history(self, learner_id):
            history = service.history_of(learner_id)
            if history is None:
                raise BadRequest(404, f"Learner {learner_id} is not in the dataset.")
            return {"learner_id": learner_id, "history": service.recent_history(history, n=len(history))}

        def learner_summary(self, learner_id):
            return {"learner_id": learner_id, "videos": len(service.history_of(learner_id))}

        def learner_recommendations(self, learner_id, query):
            k = read_k(query.get("k", [None])[0])
            history = service.history_of(learner_id)
            if history is None:
                raise BadRequest(404, f"Learner {learner_id} is not in the dataset. Use POST /recommendations instead.")
            return {"learner_id": learner_id, "recent_history": service.recent_history(history),
                    **service.recommend(history, k)}

        def history_recommendations(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                raise BadRequest(400, "The body must be JSON") from None
            ids = body.get("history", [])
            if not isinstance(ids, list) or not all(isinstance(v, int) for v in ids):
                raise BadRequest(422, "history must be a list of video ids")
            k = read_k(body.get("k"))
            history = service.indices_for(ids) if ids else np.array([], dtype=int)
            return {"recent_history": service.recent_history(history), **service.recommend(history, k)}

        def video_details(self, video_id):
            index = service.indices_for([video_id])[0]
            row = service.catalogue.items.loc[index]
            return {**service.catalogue.describe(int(index)), "theme": row["theme"],
                    "subtitles": row["description_display"]}

        def log_message(self, *args):
            pass  # keep the terminal quiet during the demo

    return Handler


def make_server(host="127.0.0.1", port=8000, service=None):
    return ThreadingHTTPServer((host, port), make_handler(service or RecommendationService()))


def main():
    parser = argparse.ArgumentParser(description="Standard-library web API for the recommender")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"Serving on http://{args.host}:{args.port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
