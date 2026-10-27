"""Apify-hosted X search (apidojo/twitter-scraper-lite) as an opt-in X backend.

Fixtures are synthetic: field names and value shapes mirror a real run of the
Actor (flat engagement counts, a nested ``author`` object, ``createdAt`` in
the legacy Twitter format) but the handles, ids and text are made up.
"""

import threading
import unittest
from unittest.mock import patch

from lib import backends, env, health, http, pipeline, schema
from lib.apify_x import (
    ACTOR_PATH,
    DEPTH_CONFIG,
    MAX_CHARGE_USD,
    _parse_tweet,
    parse_apify_response,
    search_apify,
)


def _tweet(**over):
    base = {
        "type": "tweet",
        "id": "1000000000000000001",
        "url": "https://x.com/exampledev/status/1000000000000000001",
        "twitterUrl": "https://twitter.com/exampledev/status/1000000000000000001",
        "text": "Example agents ship fast",
        "createdAt": "Fri Oct 23 08:15:50 +0000 2026",
        "likeCount": 285,
        "retweetCount": 46,
        "replyCount": 10,
        "quoteCount": 1,
        "viewCount": 57001,
        "bookmarkCount": 78,
        "isRetweet": False,
        "isReply": False,
        "isQuote": False,
        "lang": "en",
        "author": {"userName": "exampledev", "name": "Example Dev", "followers": 1200},
    }
    base.update(over)
    return base


class TestParseTweet(unittest.TestCase):
    def test_maps_the_observed_shape(self):
        item = _parse_tweet(_tweet(), 0, "example agents")
        self.assertEqual(item["id"], "AP1")
        self.assertEqual(item["author_handle"], "exampledev")
        self.assertEqual(item["url"], "https://x.com/exampledev/status/1000000000000000001")
        self.assertEqual(item["date"], "2026-10-23")
        self.assertEqual(item["text"], "Example agents ship fast")
        self.assertEqual(
            item["engagement"],
            {"likes": 285, "reposts": 46, "replies": 10, "quotes": 1, "views": 57001, "bookmarks": 78},
        )

    def test_url_is_rebuilt_from_handle_and_id_when_missing(self):
        item = _parse_tweet(_tweet(url=None, twitterUrl=None), 0, "q")
        self.assertEqual(item["url"], "https://x.com/exampledev/status/1000000000000000001")

    def test_leading_mentions_are_captured(self):
        item = _parse_tweet(_tweet(text="@alice @bob thanks for the fix"), 0, "q")
        self.assertEqual(item["mentioned_handles"], ["alice", "bob"])

    def test_missing_author_or_id_is_dropped(self):
        self.assertIsNone(_parse_tweet(_tweet(author={}), 0, "q"))
        self.assertIsNone(_parse_tweet(_tweet(id=None, url=None, twitterUrl=None), 0, "q"))

    def test_bad_counts_become_none(self):
        item = _parse_tweet(_tweet(likeCount="n/a", viewCount=None), 0, "q")
        self.assertIsNone(item["engagement"]["likes"])
        self.assertIsNone(item["engagement"]["views"])

    def test_text_is_capped(self):
        item = _parse_tweet(_tweet(text="x" * 900), 0, "q")
        self.assertEqual(len(item["text"]), 500)

    def test_parse_response_returns_items(self):
        self.assertEqual(parse_apify_response({"items": [1, 2]}), [1, 2])
        self.assertEqual(parse_apify_response({}), [])


class TestSearchApify(unittest.TestCase):
    def test_no_token_returns_error(self):
        result = search_apify("test", "2026-10-01", "2026-10-27", token="")
        self.assertEqual(result["items"], [])
        self.assertIn("APIFY_API_TOKEN", result["error"])

    @patch("lib.apify_x.http.post")
    def test_request_shape(self, mock_post):
        mock_post.return_value = [_tweet()]
        result = search_apify("example agents", "2026-10-01", "2026-10-27", depth="quick", token="tok-123")

        self.assertNotIn("error", result)
        self.assertEqual(len(result["items"]), 1)
        args, kwargs = mock_post.call_args
        url = args[0]
        self.assertTrue(
            url.endswith(f"/v2/acts/{ACTOR_PATH}/run-sync-get-dataset-items"), url
        )
        # The token travels in a header, never in the URL or the query string.
        self.assertNotIn("tok-123", url)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer tok-123")
        self.assertNotIn("tok-123", repr(kwargs["params"]))
        body = args[1] if len(args) > 1 else kwargs["json_data"]
        self.assertEqual(body["searchTerms"], ["example agents"])
        self.assertEqual(body["sort"], "Top")
        self.assertEqual(body["maxItems"], DEPTH_CONFIG["quick"]["limit"])
        self.assertEqual(body["start"], "2026-10-01")
        # The window is inclusive, so the actor's end bound is the next day.
        self.assertEqual(body["end"], "2026-10-28")
        self.assertEqual(kwargs["params"]["maxTotalChargeUsd"], MAX_CHARGE_USD)

    @patch("lib.apify_x.http.post")
    def test_a_paid_call_is_never_retried(self, mock_post):
        mock_post.return_value = []
        search_apify("example agents", "2026-10-01", "2026-10-27", token="tok")
        self.assertEqual(mock_post.call_args.kwargs["retries"], 0)

    @patch("lib.apify_x.http.post")
    def test_depth_controls_query_count_and_limit(self, mock_post):
        mock_post.return_value = []
        search_apify("latest trends in AI agents", "2026-10-01", "2026-10-27", depth="default", token="tok")
        self.assertLessEqual(mock_post.call_count, DEPTH_CONFIG["default"]["queries"])
        self.assertGreaterEqual(mock_post.call_count, 1)
        for call in mock_post.call_args_list:
            body = call.args[1] if len(call.args) > 1 else call.kwargs["json_data"]
            self.assertEqual(body["maxItems"], DEPTH_CONFIG["default"]["limit"])

    @patch("lib.apify_x.http.post")
    def test_duplicate_ids_across_queries_are_collapsed(self, mock_post):
        mock_post.return_value = [_tweet(), _tweet()]
        result = search_apify("latest trends in AI agents", "2026-10-01", "2026-10-27", depth="default", token="tok")
        self.assertEqual(len(result["items"]), 1)

    @patch("lib.apify_x.http.post")
    def test_placeholder_and_non_tweet_rows_are_skipped(self, mock_post):
        mock_post.return_value = [
            {"noResults": True},
            {"type": "mock_tweet", "id": "1", "text": "demo"},
            "not a dict",
            _tweet(),
        ]
        result = search_apify("example agents", "2026-10-01", "2026-10-27", depth="quick", token="tok")
        self.assertEqual([i["author_handle"] for i in result["items"]], ["exampledev"])

    @patch("lib.apify_x.http.post")
    def test_unexpected_payload_shape_is_an_empty_result_not_a_crash(self, mock_post):
        mock_post.return_value = {"error": {"message": "oops"}}
        result = search_apify("example agents", "2026-10-01", "2026-10-27", depth="quick", token="tok")
        self.assertEqual(result["items"], [])
        self.assertNotIn("error", result)

    @patch("lib.apify_x.http.post")
    def test_auth_failure_is_fatal_and_classified(self, mock_post):
        for status in (401, 403):
            mock_post.side_effect = http.HTTPError("Unauthorized", status_code=status)
            result = search_apify("test", "2026-10-01", "2026-10-27", token="bad")
            self.assertEqual(result["items"], [])
            self.assertIn("auth failed", result["error"])

    @patch("lib.apify_x.http.post")
    def test_payment_required_carries_the_classifier_marker(self, mock_post):
        mock_post.side_effect = http.HTTPError("Payment Required", status_code=402)
        result = search_apify("test", "2026-10-01", "2026-10-27", token="tok")
        self.assertEqual(result["items"], [])
        self.assertEqual(
            http.classify_failure(message=result["error"]), health.PAYMENT_REQUIRED
        )

    @patch("lib.apify_x.http.post")
    def test_run_timeout_is_reported(self, mock_post):
        mock_post.side_effect = http.HTTPError("Request Timeout", status_code=408)
        result = search_apify("test", "2026-10-01", "2026-10-27", token="tok")
        self.assertEqual(result["items"], [])
        self.assertEqual(http.classify_failure(message=result["error"]), health.TIMEOUT)

    @patch("lib.apify_x.http.post")
    def test_other_http_errors_are_a_quiet_empty_result(self, mock_post):
        mock_post.side_effect = http.HTTPError("Bad Gateway", status_code=502)
        result = search_apify("test", "2026-10-01", "2026-10-27", token="tok")
        self.assertEqual(result["items"], [])

    @patch("lib.apify_x.http.post")
    def test_cancelled_run_never_calls_the_api(self, mock_post):
        cancel = threading.Event()
        cancel.set()
        result = search_apify("test", "2026-10-01", "2026-10-27", token="tok", cancel=cancel)
        mock_post.assert_not_called()
        self.assertIn("cancelled or timed out", result["error"])

    @patch("lib.apify_x.http.post")
    def test_past_deadline_never_calls_the_api(self, mock_post):
        result = search_apify("test", "2026-10-01", "2026-10-27", token="tok", deadline=0.0)
        mock_post.assert_not_called()
        self.assertIn("cancelled or timed out", result["error"])


def _no_other_x_backends():
    """Hermetic chain: no cookie scraper and no xurl CLI on this machine."""
    return (
        patch("lib.bird_x.is_bird_installed", return_value=False),
        patch("lib.xurl_x.is_available", return_value=False),
        patch("lib.xurl_x.has_stored_auth", return_value=False),
    )


class TestChainRegistration(unittest.TestCase):
    def _chain(self, config):
        p1, p2, p3 = _no_other_x_backends()
        with p1, p2, p3:
            return env.x_backend_chain(config)

    def test_apify_is_opt_in_not_in_the_auto_chain(self):
        self.assertIn("apify", env.X_BACKEND_OPT_IN)
        self.assertNotIn("apify", env.X_BACKEND_ORDER)
        self.assertIn("apify", env.X_BACKEND_KNOWN)

    def test_token_alone_never_selects_apify(self):
        self.assertEqual(self._chain({"APIFY_API_TOKEN": "dummy-token"}), [])

    def test_pin_with_token_selects_apify_exclusively(self):
        config = {"APIFY_API_TOKEN": "dummy-token", "LAST30DAYS_X_BACKEND": "apify",
                  "XQUIK_API_KEY": "dummy-xquik"}
        self.assertEqual(self._chain(config), ["apify"])

    def test_pin_without_token_is_empty_not_a_fallback(self):
        config = {"LAST30DAYS_X_BACKEND": "apify", "XQUIK_API_KEY": "dummy-xquik"}
        self.assertEqual(self._chain(config), [])

    def test_source_method_label(self):
        self.assertEqual(env._X_METHOD_LABELS["apify"], "api_key")

    def test_diagnose_reports_the_pinned_backend(self):
        p1, p2, p3 = _no_other_x_backends()
        config = {"APIFY_API_TOKEN": "dummy-token", "LAST30DAYS_X_BACKEND": "apify"}
        with p1, p2, p3:
            status = env.get_x_source(config)
        self.assertEqual(status, "apify")

    def test_descriptor_is_paid_opt_in_and_key_probed(self):
        spec = next(s for s in backends.get_descriptor("x").backends if s.name == "apify")
        self.assertTrue(spec.opt_in)
        self.assertTrue(spec.paid)
        found = spec.probe({"APIFY_API_TOKEN": "dummy-token"})
        self.assertEqual(found.status, health.OK)
        missing = spec.probe({})
        self.assertNotEqual(missing.status, health.OK)

    def test_runtime_schema_accepts_the_backend(self):
        runtime = schema.ProviderRuntime(
            reasoning_provider="local", planner_model="", rerank_model="",
            x_search_backend="apify",
        )
        self.assertEqual(runtime.x_search_backend, "apify")


class TestPipelineDispatch(unittest.TestCase):
    @patch("lib.apify_x.http.post")
    def test_fetch_backend_routes_to_apify_with_the_config_token(self, mock_post):
        mock_post.return_value = [_tweet()]
        items, error = pipeline._fetch_x_backend(
            "apify", "example agents", "2026-10-01", "2026-10-27", "quick",
            {"APIFY_API_TOKEN": "dummy-token"},
        )
        self.assertEqual(error, "")
        self.assertEqual([i["author_handle"] for i in items], ["exampledev"])
        self.assertEqual(mock_post.call_args.kwargs["headers"]["Authorization"], "Bearer dummy-token")

    @patch("lib.apify_x.http.post")
    def test_payment_failure_settles_as_payment_required(self, mock_post):
        mock_post.side_effect = http.HTTPError("Payment Required", status_code=402)
        sq = schema.SubQuery(label="primary", search_query="q", ranking_query="q?", sources=["x"])
        runtime = schema.ProviderRuntime(
            reasoning_provider="mock", planner_model="mock", rerank_model="mock",
            x_search_backend=None,
        )
        p1, p2, p3 = _no_other_x_backends()
        with p1, p2, p3, self.assertRaises(pipeline.SourceRunError) as ctx:
            pipeline._retrieve_stream(
                topic="q", subquery=sq, source="x",
                config={"APIFY_API_TOKEN": "dummy-token", "LAST30DAYS_X_BACKEND": "apify"},
                depth="default", date_range=("2026-10-01", "2026-10-27"),
                runtime=runtime, mock=False,
            )
        self.assertEqual(health.PAYMENT_REQUIRED, ctx.exception.outcome_state)

    @patch("lib.apify_x.http.post")
    def test_handle_lanes_do_not_call_apify(self, mock_post):
        """Topic search only: Phase 2 lanes skip this backend (no extra paid calls)."""
        bundle = schema.RetrievalBundle()
        plan = schema.QueryPlan(
            intent="exploration", freshness_mode="balanced_recent", cluster_mode="topic",
            raw_topic="Example", source_weights={"x": 1.0},
            subqueries=[schema.SubQuery(label="primary", search_query="Example",
                                       ranking_query="Example", sources=["x"])],
        )
        from lib import providers
        config = {"APIFY_API_TOKEN": "dummy-token", "LAST30DAYS_X_BACKEND": "apify"}
        pipeline._run_supplemental_searches(
            topic="Example", bundle=bundle, plan=plan, config=config, depth="default",
            date_range=("2026-10-01", "2026-10-27"),
            runtime=providers.mock_runtime({}, "default"), mock=False,
            rate_limited_sources=set(), rate_limit_lock=threading.Lock(),
            x_handle="exampledev",
        )
        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
