"""Explicit max_tokens 0 is a value. It must not fall through to 1024."""
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "serve_openai", Path(__file__).with_name("serve_openai.py")
)
assert SPEC and SPEC.loader
serve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serve)


def _body(**extra):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return body


class ParseRequest(unittest.TestCase):
    def test_explicit_zero_stays_zero(self):
        req, err = serve.parse_request(_body(max_tokens=0))
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 0)

    def test_zero_is_not_replaced_by_max_completion_tokens(self):
        req, err = serve.parse_request(_body(max_tokens=0, max_completion_tokens=16))
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 0)

    def test_completion_token_zero_stays_zero(self):
        req, err = serve.parse_request(_body(max_completion_tokens=0))
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 0)

    def test_omitted_limit_stays_the_default(self):
        req, err = serve.parse_request(_body())
        self.assertIsNone(err)
        self.assertEqual(req["max_tokens"], 1024)

    def test_temperature_zero_stays_zero(self):
        req, err = serve.parse_request(_body(temperature=0))
        self.assertIsNone(err)
        self.assertEqual(req["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
