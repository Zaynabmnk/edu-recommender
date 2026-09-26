"""Tests for the standard-library web API. They start a real server on a free port and talk to it over HTTP."""

import json
import threading
import unittest
import urllib.error
import urllib.request

from edurec.train import MODEL_PATH


@unittest.skipUnless(MODEL_PATH.exists(), "run python -m edurec.train first")
class WebServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from edurec.webserver import make_server
        cls.server = make_server(port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_health(self):
        status, body = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body["videos"], 1167)

    def test_learner_recommendations(self):
        status, body = self.request("/learners/111316/recommendations?k=5")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["recommendations"]), 5)
        self.assertTrue(all(item["reason"] for item in body["recommendations"]))

    def test_unknown_learner_and_route(self):
        self.assertEqual(self.request("/learners/999999/recommendations")[0], 404)
        self.assertEqual(self.request("/nothing-here")[0], 404)

    def test_history_recommendations(self):
        status, body = self.request("/recommendations", {"history": [510, 511], "k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(body["history_length"], 2)
        self.assertEqual(len(body["recommendations"]), 3)
        self.assertNotIn(510, [item["item_id"] for item in body["recommendations"]])

    def test_new_learner_and_bad_input(self):
        self.assertEqual(self.request("/recommendations", {"history": []})[1]["history_group"], "new learner")
        self.assertEqual(self.request("/recommendations", {"history": [1]})[0], 422)
        self.assertEqual(self.request("/recommendations", {"history": [510], "k": 0})[0], 422)
        self.assertEqual(self.request("/recommendations", {"history": "510"})[0], 422)

    def test_demo_page_and_search(self):
        with urllib.request.urlopen(self.base + "/", timeout=10) as response:
            page = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("Next video recommender", page)
        status, body = self.request("/videos?search=pivottable&limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["videos"]), 3)
        self.assertTrue(all("pivottable" in v["name"].lower() for v in body["videos"]))

    def test_learner_history(self):
        status, body = self.request("/learners/111316/history")
        self.assertEqual(status, 200)
        self.assertEqual(len(body["history"]), 186)
        self.assertEqual(self.request("/learners/999999/history")[0], 404)

    def test_example_learners(self):
        status, body = self.request("/learners/heaviest")
        self.assertEqual((status, body["learner_id"], body["videos"]), (200, 111316, 186))
        status, body = self.request("/learners/random")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(body["videos"], 2)
        self.assertEqual(self.request(f"/learners/{body['learner_id']}/history")[0], 200)

    def test_video_details(self):
        status, body = self.request("/videos/510")
        self.assertEqual(status, 200)
        self.assertEqual(body["name"], "What is OneDrive for Business?")
        self.assertTrue(body["subtitles"].startswith("OneDrive for Business is an online library to store"))
        self.assertEqual(self.request("/videos/1")[0], 404)


if __name__ == "__main__":
    unittest.main()
