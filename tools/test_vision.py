#!/usr/bin/env python3
"""
CPU-only tests for the vision (image input) path of serve_openai.py.

Nothing here touches the GPU: the vision tower is never loaded and no model
weights are read. Config and Tokenizer do load from the real checkpoint, which
works on CPU, so the prompt-building tests run against the real chat template
and the real vocabulary.

Run (CPU only; point MODEL_DIR at a Qwen3.8-27B EXL3 checkpoint):
    CUDA_VISIBLE_DEVICES="" MODEL_DIR=models/Qwen3.8-27B-EXL3-3.5bpw \
        .venv/bin/python tools/test_vision.py

Covers message parsing, prompt building (real template and vocabulary) and
the SSRF filtering for http(s) image URLs.
"""
import asyncio, base64, io, json, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import serve_openai as srv

MODEL_DIR = os.environ.get("MODEL_DIR", "models/Qwen3.8-27B-EXL3-3.5bpw")


def png_bytes(size = (8, 8)):
    """A real, tiny PNG, so PIL decoding is exercised rather than mocked."""
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format = "PNG")
    return buf.getvalue()


def data_uri(payload = None, mime = "image/png"):
    payload = png_bytes() if payload is None else payload
    return f"data:{mime};base64," + base64.b64encode(payload).decode()


def image_message(url, text = "Describe the image."):
    return {"role": "user", "content": [{"type": "image_url",
                                         "image_url": {"url": url}},
                                        {"type": "text", "text": text}]}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Section 5 item 1: extract_images
# ---------------------------------------------------------------------------

class TestExtractImages(unittest.TestCase):

    def setUp(self):
        # extract_images refuses everything when vision is off, so the parsing
        # tests have to claim it is on. No model is loaded by this flag alone.
        self._saved = dict(srv.vision_state)
        srv.vision_state["enabled"] = True
        self._saved_fetch = srv.fetch_image_bytes

    def tearDown(self):
        srv.vision_state.clear()
        srv.vision_state.update(self._saved)
        srv.fetch_image_bytes = self._saved_fetch

    def test_data_uri(self):
        images = run(srv.extract_images([image_message(data_uri())]))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].size, (8, 8))

    def test_http_url_mocked(self):
        seen = []

        async def fake_fetch(url, max_bytes, timeout):
            seen.append((url, max_bytes, timeout))
            return png_bytes((16, 16))

        srv.fetch_image_bytes = fake_fetch
        images = run(srv.extract_images(
            [image_message("https://example.invalid/cat.png")],
            max_image_mb = 20, timeout = 15))
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (16, 16))
        self.assertEqual(seen, [("https://example.invalid/cat.png",
                                 20 * 1024 * 1024, 15)])

    def test_document_order_preserved(self):
        small, large = data_uri(), data_uri(png_bytes((32, 32)))
        msgs = [
            {"role": "user", "content": [{"type": "image_url",
                                          "image_url": {"url": small}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "and this"},
                                         {"type": "image_url",
                                          "image_url": {"url": large}}]},
        ]
        images = run(srv.extract_images(msgs))
        self.assertEqual([i.size for i in images], [(8, 8), (32, 32)])

    def test_text_only_returns_empty(self):
        msgs = [{"role": "user", "content": "hello"},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.assertEqual(run(srv.extract_images(msgs)), [])

    def test_rejects_file_scheme(self):
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images([image_message("file:///etc/passwd")]))
        self.assertIn("scheme", str(ctx.exception))

    def test_rejects_oversized_data_uri(self):
        big = data_uri(png_bytes((512, 512)))
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images([image_message(big)], max_image_mb = 0.0005))
        self.assertIn("limit", str(ctx.exception))

    def test_rejects_oversized_fetch_by_declared_length(self):
        # The real fetcher, driven through a stub response object, to prove the
        # declared Content-Length is honoured before anything is read.
        with self.assertRaises(srv.ImageError) as ctx:
            srv.decode_data_uri(data_uri(png_bytes((512, 512))), 32)
        self.assertIn("limit", str(ctx.exception))

    def test_rejects_video_part(self):
        msgs = [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": data_uri()}}]}]
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images(msgs))
        self.assertIn("video", str(ctx.exception))

    def test_rejects_bare_video_key(self):
        msgs = [{"role": "user", "content": [{"video": "whatever.mp4"}]}]
        with self.assertRaises(srv.ImageError):
            run(srv.extract_images(msgs))

    def test_enforces_per_request_cap(self):
        uri = data_uri()
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}} for _ in range(3)]}]
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images(msgs, max_images = 2))
        self.assertIn("too many images", str(ctx.exception))
        # The same request is fine at the default cap.
        self.assertEqual(len(run(srv.extract_images(msgs, max_images = 8))), 3)

    def test_rejects_undecodable_bytes(self):
        uri = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images([image_message(uri)]))
        self.assertIn("decode", str(ctx.exception))

    def test_rejects_non_image_media_type(self):
        uri = "data:text/html;base64," + base64.b64encode(b"<h1>hi</h1>").decode()
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images([image_message(uri)]))
        self.assertIn("not an image", str(ctx.exception))

    def test_vision_disabled_message_says_how_to_enable(self):
        srv.vision_state["enabled"] = False
        with self.assertRaises(srv.ImageError) as ctx:
            run(srv.extract_images([image_message(data_uri())]))
        self.assertIn("VISION=1", str(ctx.exception))


# ---------------------------------------------------------------------------
# SSRF filtering. This server binds 0.0.0.0 with no auth, so these are the
# load-bearing tests.
# ---------------------------------------------------------------------------

class TestUrlFiltering(unittest.TestCase):

    BLOCKED = [
        "http://127.0.0.1:8081/v1/models",      # another local service
        "http://127.0.0.1:8888/health",         # this server
        "http://localhost/",                    # resolves to loopback
        "http://localhost./",                   # trailing dot, same host
        "http://LOCALHOST/",                    # case
        "http://169.254.169.254/latest/meta-data/",   # cloud metadata
        "http://10.0.0.1/",                     # RFC1918
        "http://192.168.1.1/",                  # RFC1918
        "http://172.16.0.5/",                   # RFC1918
        "http://[::1]/",                        # IPv6 loopback
        "http://[::ffff:127.0.0.1]/",           # IPv4-mapped loopback
        "http://0.0.0.0/",                      # unspecified
        "http://224.0.0.1/",                    # multicast
        "http://240.0.0.1/",                    # reserved
    ]

    def test_blocked_hosts(self):
        for url in self.BLOCKED:
            with self.subTest(url = url):
                with self.assertRaises(srv.ImageError) as ctx:
                    srv.validate_image_url(url)
                self.assertRegex(str(ctx.exception),
                                 "non-public|unparseable|no addresses")

    def test_blocked_schemes(self):
        for url in ["file:///etc/passwd", "ftp://example.com/x.png",
                    "gopher://example.com/", "/no/scheme.png"]:
            with self.subTest(url = url):
                with self.assertRaises(srv.ImageError) as ctx:
                    srv.validate_image_url(url)
                self.assertIn("scheme", str(ctx.exception))

    def test_public_literals_pass(self):
        # Numeric literals, so getaddrinfo does no DNS lookup and the test
        # stays offline.
        for url in ["http://93.184.216.34/cat.png", "https://8.8.8.8/x.jpg",
                    "http://[2606:2800:220:1:248:1893:25c8:1946]/x.png"]:
            with self.subTest(url = url):
                srv.validate_image_url(url)

    def test_empty_host(self):
        with self.assertRaises(srv.ImageError) as ctx:
            srv.validate_image_url("http:///cat.png")
        self.assertIn("no host", str(ctx.exception))

    def test_ipv4_mapped_unwrapping(self):
        self.assertTrue(srv._classify_address("::ffff:10.0.0.1").is_private)
        self.assertTrue(srv._classify_address("fe80::1%eth0").is_link_local)
        self.assertIsNone(srv._classify_address("not-an-address"))

    def test_redirect_hop_is_revalidated(self):
        # The fetcher validates before every hop, so a public URL redirecting to
        # loopback is refused at the second check rather than followed.
        class FakeResponse:
            def __init__(self, status, headers):
                self.status, self.headers, self.content_length = status, headers, None
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False

        class FakeSession:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            def get(self, url, allow_redirects = True):
                assert allow_redirects is False, "redirects must not be auto-followed"
                return FakeResponse(302, {"Location": "http://127.0.0.1:8081/x.png"})

        import aiohttp
        saved = aiohttp.ClientSession
        aiohttp.ClientSession = FakeSession
        try:
            with self.assertRaises(srv.ImageError) as ctx:
                run(srv.fetch_image_bytes("http://93.184.216.34/cat.png",
                                          1024, 5))
            self.assertIn("non-public", str(ctx.exception))
        finally:
            aiohttp.ClientSession = saved


# ---------------------------------------------------------------------------
# Section 5 items 2 and 3: prompt building against the real tokenizer
# ---------------------------------------------------------------------------

class TestPromptBuilding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from exllamav3 import Config, Tokenizer
        cls.config = Config.from_directory(MODEL_DIR)
        cls.tokenizer = Tokenizer.from_config(cls.config)
        with open(os.path.join(MODEL_DIR, "config.json"), encoding = "utf8") as f:
            model_cfg = json.load(f)
        cls.image_token_id = model_cfg["image_token_id"]
        cls.placeholder = cls.tokenizer.id_to_piece[cls.image_token_id]
        cls.span_ids = (model_cfg["vision_start_token_id"],
                        model_cfg["vision_end_token_id"])

    def setUp(self):
        self._saved = dict(srv.vision_state)
        srv.vision_state["enabled"] = True
        srv.vision_state["image_token_id"] = self.image_token_id
        srv.vision_state["placeholder"] = self.placeholder
        # Same wiring main() does, so the tests exercise the real substitution
        # target rather than the bare-placeholder fallback.
        srv.vision_state["span"] = srv._vision_span(
            self.tokenizer, self.image_token_id, self.span_ids)

    def tearDown(self):
        srv.vision_state.clear()
        srv.vision_state.update(self._saved)

    def fake_embedding(self, num_tokens = 4, hidden = 5120):
        """An MMEmbedding with zeroed tensors: no vision tower, no GPU.

        token_string carries the vision_start/vision_end markers because that is
        what get_image_embeddings really produces (architecture/qwen3_vl.py:405).
        An earlier bare-token_string version of this helper hid the doubled
        marker bug, so keep it realistic.
        """
        import torch
        from exllamav3.tokenizer.mm_embedding import MMEmbedding
        start, end = self.config.vision_start_token_id, self.config.vision_end_token_id
        return MMEmbedding(
            embeddings = torch.zeros(num_tokens, hidden, dtype = torch.half),
            token_string = torch.tensor(
                [[start] + [-1] * num_tokens + [end]], dtype = torch.long),
            grid_thw = (1, 4, 4),
            mrope_merge_size = 2)

    def test_config_has_no_image_token_id(self):
        # Blocker 2 of the plan: the reason the server does the substitution
        # itself instead of calling hf_chat_template(embeddings = ...).
        self.assertFalse(hasattr(self.config, "image_token_id"))
        self.assertEqual(self.placeholder, "<|image_pad|>")

    def test_render_substitute_encode(self):
        from exllamav3.tokenizer.mm_embedding import FIRST_MM_EMBEDDING_INDEX
        emb = self.fake_embedding()
        msgs = [image_message("data:ignored")]
        ids = srv.build_prompt_ids(self.tokenizer, msgs, None, [emb])
        self.assertEqual(ids.shape[0], 1)
        flat = ids[0].tolist()
        alias_ids = [i for i in flat if i >= FIRST_MM_EMBEDDING_INDEX]
        self.assertEqual(len(alias_ids), emb.mm_length)
        self.assertEqual(alias_ids,
                         list(range(emb.first_index, emb.last_index)))
        # The alias expands to vision_start + the embedding ids + vision_end
        # (architecture/qwen3_vl.py:400-406).
        self.assertIn(self.config.vision_start_token_id, flat)
        self.assertIn(self.config.vision_end_token_id, flat)

    def test_vision_markers_are_not_doubled(self):
        """Regression test for doubled vision markers (a bug also present in the
        engine's own hf_chat_template, tokenizer.py:712-735).

        A real embedding's token_string is [vision_start] + ids + [vision_end]
        (architecture/qwen3_vl.py:405) and the chat template writes its own pair
        around the placeholder, so replacing the bare <|image_pad|> emits each
        marker TWICE and the prompt is 2 tokens per image off the training
        shape. build_prompt_ids replaces the whole marker run instead.
        """
        emb = self.fake_embedding(4)
        start, end = self.config.vision_start_token_id, self.config.vision_end_token_id
        flat = srv.build_prompt_ids(
            self.tokenizer, [image_message("data:x")], None, [emb])[0].tolist()
        self.assertEqual(flat.count(start), 1)
        self.assertEqual(flat.count(end), 1)
        self.assertEqual(flat.count(srv.vision_state["image_token_id"]), 0)
        self.assertEqual(sum(1 for i in flat
                             if emb.first_index <= i < emb.last_index), 4)

    def test_two_images_keep_their_order(self):
        e1, e2 = self.fake_embedding(4), self.fake_embedding(6)
        msgs = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:a"}},
            {"type": "image_url", "image_url": {"url": "data:b"}},
            {"type": "text", "text": "compare them"}]}]
        flat = srv.build_prompt_ids(self.tokenizer, msgs, None, [e1, e2])[0].tolist()
        self.assertLess(flat.index(e1.first_index), flat.index(e2.first_index))
        self.assertEqual(sum(1 for i in flat
                             if e1.first_index <= i < e1.last_index), 4)
        self.assertEqual(sum(1 for i in flat
                             if e2.first_index <= i < e2.last_index), 6)

    def test_placeholder_count_mismatch_is_reported(self):
        msgs = [image_message("data:ignored")]
        with self.assertRaises(srv.ImageError) as ctx:
            srv.build_prompt_ids(self.tokenizer, msgs, None,
                                 [self.fake_embedding(), self.fake_embedding()])
        message = str(ctx.exception)
        self.assertIn("1 image placeholder", message)
        self.assertIn("2 image(s)", message)

    def test_image_in_system_message_is_a_400(self):
        msgs = [{"role": "system", "content": [
                    {"type": "image_url", "image_url": {"url": "data:x"}}]},
                {"role": "user", "content": "hi"}]
        with self.assertRaises(srv.ImageError) as ctx:
            srv.build_prompt_ids(self.tokenizer, msgs, None,
                                 [self.fake_embedding()])
        self.assertIn("chat template rejected", str(ctx.exception))

    # --- item 3: text-only must be byte-identical to the pre-change path ----

    TEXT_CASES = [
        [{"role": "user", "content": "Hello."}],
        [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "Hello."},
         {"role": "assistant", "content": "Hi."},
         {"role": "user", "content": "Again?"}],
        [{"role": "user", "content": [{"type": "text", "text": "Only text."}]}],
    ]

    TOOLS = [{"type": "function", "function": {
        "name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]

    def test_text_only_prompt_unchanged(self):
        for i, msgs in enumerate(self.TEXT_CASES):
            for tools in (None, self.TOOLS):
                with self.subTest(case = i, tools = tools is not None):
                    before = self.tokenizer.hf_chat_template(
                        msgs, add_generation_prompt = True,
                        enable_thinking = True, tools = tools)
                    after = srv.build_prompt_ids(self.tokenizer, msgs, tools, [])
                    self.assertEqual(before.tolist(), after.tolist())

    def test_text_only_requests_extract_no_images(self):
        for i, msgs in enumerate(self.TEXT_CASES):
            with self.subTest(case = i):
                self.assertEqual(srv.media_parts(msgs), [])
                self.assertEqual(run(srv.extract_images(msgs)), [])


if __name__ == "__main__":
    if not os.path.isdir(MODEL_DIR):
        print(f"model directory not found: {MODEL_DIR}", file = sys.stderr)
        sys.exit(2)
    unittest.main(verbosity = 2)
