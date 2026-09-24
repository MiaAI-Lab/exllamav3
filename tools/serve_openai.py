#!/usr/bin/env python3
"""
Minimal OpenAI-compatible server for the EXL3 serving target.

Drafter: MTP by default (`-dm mtp`; the draft head lives inside the target
checkpoint). Alternatives: a DFlash2 draft model directory (`-dm <dir>`),
or no drafting (`-dm none`). The start.sh launcher maps the .env `DRAFT`
knob onto these.

Endpoints:
  GET  /v1/models
  GET  /health
  POST /v1/chat/completions   (stream and non-stream, tool calling)

Defaults match the serving convention: temperature 0.6, top-k 20, top-p 0.95,
thinking enabled (reasoning arrives inline in `<think>`), speculative
drafting active (drafter chosen via -dm, see above).
Concurrency: requests are serialized (batch-1 draft); concurrent callers queue.

Vision (image input, optional, off by default):
  - `--vision` loads the checkpoint's vision component and accepts OpenAI image
    parts, `{"type": "image_url", "image_url": {"url": ...}}`, carrying either a
    `data:` URI or an http(s) URL.
  - Remote URLs are resolved and refused if they point at loopback, private,
    link-local, multicast, reserved or unspecified addresses (this server has no
    authentication, so an unfiltered fetcher would be an open proxy into the
    host's network). Redirects are re-validated hop by hop.
  - Video input is rejected.

Tool calling (Qwen3.8 XML format):
  - `tools` (OpenAI function specs) are rendered by the model's HF chat template
    (system "# Tools" section). `tool_choice` is accepted; required/specific
    choices are enforced with an explicit system directive.
  - assistant history with `tool_calls` is re-rendered natively by the template
    (arguments are converted JSON-string -> dict, as the template expects).
  - `role:"tool"` messages render as `<tool_response>` blocks natively.
  - Model output `<tool_call><function=name><parameter=k>v</parameter>
    </function></tool_call>` is parsed back into OpenAI `tool_calls` objects;
    generation stops at `</tool_call>`, finish_reason = "tool_calls".
  - Tool-call arguments are typed per the request's own JSON schemas
    (integer/number/boolean/array/object), strings kept on mismatch.

Launch (from repo root):
  .venv/bin/python tools/serve_openai.py \
      -m models/Qwen3.8-27B-EXL3-3.5bpw -gs 22 -cs 262144 -cq nvfp4 --port 8888
"""
import argparse, base64, binascii, io, ipaddress, json, os, re, socket, sys, \
       time, threading, uuid
from urllib.parse import urljoin, urlparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aiohttp import web

MODEL_DIR = "test_models/Qwen3.8-27B-exl3-3.5bpw-wm"
DRAFT_DIR = "mtp"   # default drafting method: MTP head (no external draft model)
PORT = 8888

gen_lock = threading.Lock()          # serialize generation (batch-1 draft)
stats_lock = threading.Lock()
# Cumulative counters for sparkDash live tok/s (GET /health).
stats = {
    "prompt_tokens_total": 0,
    "completion_tokens_total": 0,
    "context_length": None,
}

def _bump_stats(prompt=0, completion=0):
    if prompt <= 0 and completion <= 0:
        return
    with stats_lock:
        if prompt > 0:
            stats["prompt_tokens_total"] += int(prompt)
        if completion > 0:
            stats["completion_tokens_total"] += int(completion)

def _result_new_tokens(r):
    ids = r.get("token_ids") if isinstance(r, dict) else None
    if ids is None:
        return 0
    try:
        return int(ids.shape[-1])
    except Exception:
        return 0

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
HOLD_BACK = 16                       # marker-safe holdback for streamed text

# ---------------------------------------------------------------------------
# Vision (image input)
# ---------------------------------------------------------------------------
# Filled in by main() when --vision is given; every image path reads it. Kept
# as module state (like `stats`) so the request handlers do not have to thread
# the vision model through every call.
vision_state = {
    "enabled": False,
    "model": None,              # vision component model, loaded before the text model
    "image_token_id": None,     # read from config.json, not from the Config object
    "placeholder": None,        # id_to_piece[image_token_id], i.e. "<|image_pad|>"
    "span": None,               # vision_start + placeholder + vision_end, when both ids exist
    "max_images": 8,
    "max_image_mb": 20,
    "fetch_timeout": 15,
}

MAX_IMAGE_REDIRECTS = 3         # hops allowed per image URL, each re-validated


class ImageError(Exception):
    """Bad image input. Always reported to the client as HTTP 400."""


def _classify_address(text):
    """One resolved address -> ipaddress object, IPv4-mapped forms unwrapped.
    Returns None if it cannot be parsed, which callers treat as a rejection."""
    addr = text.split("%", 1)[0]        # strip an IPv6 zone id, e.g. fe80::1%eth0
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return None
    # ::ffff:127.0.0.1 is loopback wearing an IPv6 literal; classify the inner
    # address so the checks below cannot be walked around with a v6 spelling.
    mapped = getattr(ip, "ipv4_mapped", None)
    return mapped if mapped is not None else ip


def validate_image_url(url):
    """Resolve an http(s) image URL's host and refuse anything pointing back into
    this machine or the local network. Returns the resolved addresses.

    This server binds 0.0.0.0 with no authentication, so an unfiltered fetcher is
    an open proxy into whatever the host can reach: this server itself, any other
    local service, and 169.254.169.254 (the cloud metadata address).

    It does not close a DNS rebind (aiohttp resolves the name again when it
    connects); it closes the direct and redirect-driven cases.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ImageError(f"unsupported image URL scheme {parsed.scheme or '(none)'!r}: "
                         "only data:, http: and https: are accepted")
    # rstrip("."): "localhost." is the same host to the resolver but a different
    # string to any comparison done before resolving.
    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        raise ImageError("image URL has no host")
    try:
        infos = socket.getaddrinfo(host, None, proto = socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ImageError(f"cannot resolve image URL host {host!r}: {e}")
    if not infos:
        raise ImageError(f"image URL host {host!r} resolved to no addresses")
    addrs = []
    for info in infos:
        ip = _classify_address(info[4][0])
        if ip is None:
            raise ImageError(f"image URL host {host!r} resolved to an "
                             f"unparseable address {info[4][0]!r}")
        if (ip.is_loopback or ip.is_private or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            raise ImageError(f"image URL host {host!r} resolves to a non-public "
                             f"address ({ip}); refusing to fetch it")
        addrs.append(str(ip))
    return addrs


async def fetch_image_bytes(url, max_bytes, timeout):
    """Fetch one http(s) image, with the host check applied to every hop."""
    import aiohttp
    current = url
    hops = 0
    client_timeout = aiohttp.ClientTimeout(total = timeout)
    try:
        async with aiohttp.ClientSession(timeout = client_timeout) as session:
            while True:
                validate_image_url(current)
                # allow_redirects = False: left on, aiohttp would follow a
                # redirect into 127.0.0.1 without the check above ever seeing it.
                async with session.get(current, allow_redirects = False) as resp:
                    if resp.status in (301, 302, 303, 307, 308):
                        hops += 1
                        if hops > MAX_IMAGE_REDIRECTS:
                            raise ImageError(f"image URL redirected more than "
                                             f"{MAX_IMAGE_REDIRECTS} times")
                        location = resp.headers.get("Location")
                        if not location:
                            raise ImageError("image URL returned a redirect with "
                                             "no Location header")
                        current = urljoin(current, location)
                        continue
                    if resp.status != 200:
                        raise ImageError(f"image URL returned HTTP {resp.status}")
                    declared = resp.content_length
                    if declared is not None and declared > max_bytes:
                        raise ImageError(f"image is {declared} bytes, over the "
                                         f"{max_bytes} byte limit")
                    data = b""
                    # Read incrementally: a lying or absent Content-Length must
                    # not let an endless response fill memory.
                    async for chunk in resp.content.iter_chunked(65536):
                        data += chunk
                        if len(data) > max_bytes:
                            raise ImageError(f"image exceeds the {max_bytes} byte limit")
                    return data
    except ImageError:
        raise
    except (TimeoutError, aiohttp.ServerTimeoutError):
        raise ImageError(f"fetching the image timed out after {timeout}s")
    except aiohttp.ClientError as e:
        raise ImageError(f"could not fetch the image: {e}")


def decode_data_uri(uri, max_bytes):
    """data: URI -> raw bytes, with the size cap applied before decoding."""
    header, comma, payload = uri[len("data:"):].partition(",")
    if not comma:
        raise ImageError("malformed data: URI (no comma separating the payload)")
    params = [p.strip().lower() for p in header.split(";")]
    mime = params[0]
    if "base64" not in params[1:]:
        raise ImageError("data: URI must be base64 encoded")
    if mime and not mime.startswith("image/"):
        raise ImageError(f"data: URI media type {mime!r} is not an image")
    payload = "".join(payload.split())        # JSON bodies often wrap the base64
    # 4 base64 characters per 3 bytes: refuse before allocating the decode buffer
    if len(payload) // 4 * 3 > max_bytes:
        raise ImageError(f"image exceeds the {max_bytes} byte limit")
    try:
        data = base64.b64decode(payload, validate = True)
    except (binascii.Error, ValueError) as e:
        raise ImageError(f"could not decode base64 image data: {e}")
    if len(data) > max_bytes:
        raise ImageError(f"image exceeds the {max_bytes} byte limit")
    return data


def decode_image(data):
    """Raw bytes -> RGB PIL image (what the vision tower's preprocess expects)."""
    # Imported here, not at module scope, so a text-only deployment never needs
    # pillow present just to start the server.
    from PIL import Image
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except ImageError:
        raise
    except Exception as e:
        raise ImageError(f"could not decode image: {e}")


def _is_video_part(part):
    return "video" in part or "video_url" in part or part.get("type") in ("video", "video_url")


def _is_image_part(part):
    return "image" in part or "image_url" in part or part.get("type") in ("image", "image_url")


def _part_url(part):
    """Pull the URL out of an OpenAI image part, in either accepted shape."""
    value = part.get("image_url", part.get("image"))
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value.strip():
        raise ImageError("image part carries no URL "
                         "(expected image_url.url or image as a string)")
    return value.strip()


def media_parts(messages):
    """Every image-ish or video-ish content part, in document order.

    Matches what the chat template recognises (chat_template.jinja:8-30): an
    image/image_url/video/video_url key, or a `type` field naming one. Order is
    load-bearing: the template emits one placeholder per part in this same
    order, and build_prompt_ids() pairs them up positionally.
    """
    parts = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and (_is_image_part(part) or _is_video_part(part)):
                parts.append(part)
    return parts


async def extract_images(messages, max_images = None, max_image_mb = None,
                         timeout = None):
    """OpenAI content arrays -> list of RGB PIL images, in document order.

    Returns [] for text-only requests, which keeps their path untouched. Every
    failure raises ImageError, which the handlers turn into a 400.
    """
    max_images = vision_state["max_images"] if max_images is None else max_images
    max_image_mb = vision_state["max_image_mb"] if max_image_mb is None else max_image_mb
    timeout = vision_state["fetch_timeout"] if timeout is None else timeout
    max_bytes = int(max_image_mb * 1024 * 1024)

    parts = media_parts(messages)
    if not parts:
        return []
    if not vision_state["enabled"]:
        raise ImageError("image input is disabled on this server "
                         "(set VISION=1 in .env and restart)")
    if len(parts) > max_images:
        raise ImageError(f"too many images in one request: {len(parts)} "
                         f"(limit {max_images})")

    images = []
    for part in parts:
        # Checked before the image branch: a video part would otherwise reach the
        # template, which emits a <|video_pad|> with no embedding behind it.
        if _is_video_part(part):
            raise ImageError("video input is not supported")
        url = _part_url(part)
        if url.startswith("data:"):
            data = decode_data_uri(url, max_bytes)
        else:
            data = await fetch_image_bytes(url, max_bytes, timeout)
        images.append(decode_image(data))
    return images


def embed_images(tokenizer, images):
    """PIL images -> MMEmbeddings. Runs a GPU forward, so callers hold gen_lock."""
    model = vision_state["model"]
    if model is None:
        raise ImageError("image input is disabled on this server "
                         "(set VISION=1 in .env and restart)")
    return [model.get_image_embeddings(tokenizer = tokenizer, image = image)
            for image in images]


def _vision_span(tokenizer, image_token_id, vision_span_ids):
    """The full marker run the chat template writes around one image, e.g.
    "<|vision_start|><|image_pad|><|vision_end|>", or None if this model's
    config does not declare the surrounding ids.

    This is what must be replaced by an embedding alias, NOT the bare
    placeholder: get_image_embeddings builds token_string as
    [vision_start] + ids + [vision_end] (architecture/qwen3_vl.py:405), so the
    alias already carries the markers. Replacing only the placeholder leaves the
    template's own pair in place and the prompt ends up with each marker twice.
    """
    start_id, end_id = vision_span_ids
    if start_id is None or end_id is None:
        return None
    vocab = len(tokenizer.id_to_piece)
    if not (0 <= start_id < vocab and 0 <= end_id < vocab):
        return None
    return (tokenizer.id_to_piece[start_id]
            + tokenizer.id_to_piece[image_token_id]
            + tokenizer.id_to_piece[end_id])


def build_prompt_ids(tokenizer, messages, tools, embeddings):
    """Render the chat template and tokenize, substituting the template's image
    placeholders for embedding aliases when there are images.

    The engine's own `hf_chat_template(..., embeddings = ...)` does exactly this
    (tokenizer/tokenizer.py:712-735) but reads `config.image_token_id`, which
    only the gemma4, muse_glimmer and step3_7 config classes set. Qwen3_5VLConfig
    does not, so that call raises AttributeError here. Doing the three steps in
    the server keeps the engine unforked for this.
    """
    if not embeddings:
        return tokenizer.hf_chat_template(
            messages, add_generation_prompt = True, enable_thinking = True,
            tools = tools)

    placeholder = vision_state["placeholder"]
    try:
        rendered = tokenizer.hf_render_chat_template(
            messages, add_generation_prompt = True, enable_thinking = True,
            tools = tools)
    except Exception as e:
        # The template raises on an image inside a system message, among other
        # shapes it refuses; surface that as a 400 rather than a 500.
        raise ImageError(f"chat template rejected this request: {e}")
    # Replace the whole vision_start/image_pad/vision_end run, not just the
    # placeholder: the alias expands to [vision_start] + ids + [vision_end]
    # (architecture/qwen3_vl.py:405), so replacing the placeholder alone would
    # leave the template's own markers around it and emit each one twice. The
    # engine's own hf_chat_template has this bug; we do not copy it. Models whose
    # config omits the surrounding ids fall back to the bare placeholder.
    target = vision_state["span"] or placeholder
    if vision_state["span"] and placeholder in rendered.replace(vision_state["span"], "", len(embeddings)):
        # A placeholder outside a full marker run means the template shape is not
        # what this substitution assumes; refuse rather than build a bad prompt.
        raise ImageError("chat template emitted an image placeholder outside a "
                         "vision_start/vision_end pair; refusing to build the prompt")
    count = rendered.count(target)
    if count != len(embeddings):
        raise ImageError(f"chat template rendered {count} image placeholder(s) "
                         f"but {len(embeddings)} image(s) were supplied")
    for embedding in embeddings:
        rendered = rendered.replace(target, embedding.text_alias, 1)
    return tokenizer.encode(rendered, encode_special_tokens = True,
                            embeddings = embeddings)


def build_model(argv, use_draft = True, use_vision = False):
    from argparse import ArgumentParser

    # The one-time JIT build of the CUDA extension can look like a hang;
    # say so before the import below blocks on it.
    try:
        import importlib.util, os
        if importlib.util.find_spec("exllamav3_ext") is None:
            _root = os.environ.get("TORCH_EXTENSIONS_DIR",
                                   os.path.expanduser("~/.cache/torch_extensions"))
            if not (os.path.isdir(_root) and
                    any(d == "exllamav3_ext"
                        for _, _dirs, _ in os.walk(_root) for d in _dirs)):
                print(" == compiling the CUDA extension "
                      "(one-time; a few minutes of silence is normal) ...", flush = True)
    except Exception:
        pass

    from exllamav3 import model_init, Generator
    parser = ArgumentParser()
    model_init.add_args(parser, add_draft_model_args = use_draft)
    args = parser.parse_args(argv)

    vision_model, image_token_id, vision_span_ids = None, None, (None, None)
    if use_vision:
        from exllamav3 import Config, Model
        vision_config = Config.from_directory(args.model_dir)
        if vision_config.vision is None or "vision" not in vision_config.model_classes:
            raise SystemExit(f" !! --vision: {args.model_dir} has no vision component "
                             "(no vision_config in its config.json)")
        # image_token_id comes straight from config.json: Qwen3_5VLConfig never
        # copies it onto the Config object (see build_prompt_ids).
        with open(os.path.join(args.model_dir, "config.json"), encoding = "utf8") as f:
            _model_cfg = json.load(f)
        image_token_id = _model_cfg.get("image_token_id")
        vision_span_ids = (_model_cfg.get("vision_start_token_id"),
                           _model_cfg.get("vision_end_token_id"))
        if image_token_id is None:
            raise SystemExit(f" !! --vision: {args.model_dir}/config.json has no "
                             "image_token_id, so image placeholders cannot be located")
        # Loaded BEFORE the text model on purpose. The text autosplit sets a
        # process-wide memory fraction from -gs, and the vision load's measuring
        # forward uses a large dummy grid (11008 patches), so it wants room of its
        # own; loading it first lets the text split adapt around it. This is the
        # order examples/multimodal.py uses too.
        print(f" == loading vision component of {args.model_dir} ...", flush = True)
        vision_model = Model.from_config(vision_config, component = "vision")
        vision_model.load(progressbar = True)

    if use_draft:
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(args, progress = True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model = draft_model, draft_cache = draft_cache,
            # num_draft_tokens defaults to the draft model's arch-declared
            # default_draft_size (DFlash2: block_size - 1 = 7, MTP: 4). Must
            # match model_init's max_history sizing, which reads the same caps.
        )
    else:
        model, config, cache, tokenizer = model_init.init(args, progress = True)
        generator = Generator(model, cache, tokenizer)
    return generator, tokenizer, vision_model, image_token_id, vision_span_ids


def normalize_messages(messages):
    """OpenAI history -> template-compatible dicts (tool_calls args str->dict)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                fn["arguments"] = args
                calls.append({"function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def split_reasoning(text):
    """Split Qwen reasoning from content. Generation starts inside <think>
    (the chat template ends with it), so text before </think> is reasoning.
    Returns (reasoning, content) with markers stripped."""
    close = text.find("</think>")
    if close >= 0:
        reasoning = text[:close]
        content = text[close + len("</think>"):]
        return reasoning.lstrip().removeprefix("<think>").strip(), content.strip("\n")
    if text.lstrip().startswith("<think>"):
        return text.lstrip()[len("<think>"):].strip(), ""
    return "", text


def build_tool_schemas(tools):
    """OpenAI tools list -> {function_name: {param_name: json-schema type}}."""
    schemas = {}
    for t in tools or []:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if name and isinstance(props, dict):
            schemas[name] = {k: v.get("type") for k, v in props.items()
                             if isinstance(v, dict)}
    return schemas


def _coerce_value(value, jtype):
    """Coerce one XML string parameter to the schema-declared JSON type.
    Lossless: on any mismatch the original string is returned unchanged."""
    v = value.strip()
    if not v:
        return value
    try:
        if jtype == "integer":
            return int(v)
        if jtype == "number":
            try:
                return int(v)
            except ValueError:
                return float(v)
        if jtype == "boolean":
            if v.lower() == "true": return True
            if v.lower() == "false": return False
        if jtype == "array":
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return parsed
        if jtype == "object":
            parsed = json.loads(v)
            if isinstance(parsed, dict):
                return parsed
    except (ValueError, json.JSONDecodeError):
        pass
    return value


def coerce_tool_args(args, fn_schema):
    """Qwen's XML tool format delivers every parameter value as a string;
    OpenAI tool_calls arguments are typed JSON. Coerce each value using the
    request's own tool schema; undeclared params and failed coercions keep
    the raw string."""
    if not fn_schema:
        return args
    out = {}
    for k, v in args.items():
        t = fn_schema.get(k)
        types = t if isinstance(t, list) else [t]
        for tt in types:
            if isinstance(tt, str) and tt in ("integer", "number", "boolean",
                                              "array", "object"):
                cv = _coerce_value(v, tt)
                if not isinstance(cv, str):
                    v = cv
                    break
        out[k] = v
    return out


def parse_tool_calls(text, tool_schemas = None):
    """Parse Qwen XML tool calls. Returns (content_without_calls, [calls]).
    A <tool_call> block left unterminated is treated as complete: the
    </tool_call> stop-condition strips the closing tag from generated text."""
    calls = []
    content = text

    def parse_block(block):
        fm = re.search(r"<function=([^>]+)>", block)
        if not fm:
            return None
        name = fm.group(1).strip()
        args = {}
        for pm in re.finditer(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>",
                              block[fm.end():], flags = re.S):
            args[pm.group(1).strip()] = pm.group(2)
        if tool_schemas:
            args = coerce_tool_args(args, tool_schemas.get(name))
        return {
            "id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    while True:
        i = content.find(TOOL_CALL_OPEN)
        if i < 0:
            break
        j = content.find(TOOL_CALL_CLOSE, i)
        if j < 0:
            # truncated close (stop string consumed): parse the remainder
            call = parse_block(content[i + len(TOOL_CALL_OPEN):])
            if call:
                calls.append(call)
            content = content[:i]
            break
        call = parse_block(content[i + len(TOOL_CALL_OPEN):j])
        if call:
            calls.append(call)
        content = content[:i] + content[j + len(TOOL_CALL_CLOSE):]
    return content, calls


def tool_choice_directive(tool_choice, tools):
    """OpenAI tool_choice -> (tools_to_render, extra system directive or None).
    The Qwen template has no tool_choice support, so required/specific are
    enforced with an explicit instruction appended to the history."""
    if tool_choice in (None, "auto"):
        return tools, None
    if tool_choice == "none":
        return None, None
    names = [t["function"]["name"] for t in (tools or [])
             if isinstance(t, dict) and t.get("type") == "function"]
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        return tools, (f"You must call the function `{name}` now. Reply ONLY with "
                       f"the <tool_call> block for `{name}` and nothing else.")
    if tool_choice == "required":
        one_of = " or ".join(f"`{n}`" for n in names)
        return tools, (f"You must call one of the available functions ({one_of}) "
                       "now. Reply ONLY with the <tool_call> block and nothing else.")
    return tools, None


# Per-reply ceiling when the client sends no max_tokens. Covers long thinking;
# only a stuck, looping reply gets near it. Override with env DEFAULT_MAX_TOKENS.
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "32768"))


def generate_full(generator, tokenizer, messages, max_tokens, temperature,
                  top_p, top_k, seed, tools, tool_choice = None, stop = None,
                  on_text = None, images = None):
    """Blocking generation; returns (text, tool_calls, finish, p_toks, o_toks,
    reasoning, content)."""
    schemas = build_tool_schemas(tools)
    tools, directive = tool_choice_directive(tool_choice, tools)
    if directive:
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            # Qwen template allows only ONE leading system message — merge
            first = dict(messages[0])
            first["content"] = (first.get("content") or "").rstrip() + "\n\n" + directive
            messages[0] = first
        else:
            messages = [{"role": "system", "content": directive}] + messages
    from exllamav3.generator.sampler.presets import ComboSampler
    from exllamav3 import Job
    forced_choice = tool_choice not in (None, "auto", "none")
    reason = "max_new_tokens"
    text = ""
    embeddings = []
    input_ids = None
    prompt_toks = 0

    def run_once():
        nonlocal text, reason
        text = ""
        reason = "max_new_tokens"
        sampler = ComboSampler(temperature = temperature, top_k = top_k, top_p = top_p)
        stop_conditions = ["<|im_end|>", tokenizer.eos_token_id] + (stop or [])
        # embeddings must reach the Job as well as the input ids: the prefill
        # path forwards them as indexed_embeddings (job.py:65,1237).
        job = Job(input_ids = input_ids, max_new_tokens = max_tokens,
                  stop_conditions = stop_conditions,
                  sampler = sampler, seed = seed,
                  embeddings = embeddings or None)
        prefill_seen = 0
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("stage") == "prefill":
                    curr = int(r.get("curr_progress") or 0)
                    if curr > prefill_seen:
                        _bump_stats(prompt=curr - prefill_seen)
                        prefill_seen = curr
                elif _result_new_tokens(r):
                    _bump_stats(completion=_result_new_tokens(r))
                chunk = r.get("text", "")
                if chunk:
                    text += chunk
                    if on_text is not None:
                        on_text(chunk)
                if r.get("eos"):
                    reason = r.get("eos_reason", reason)
        if prefill_seen < prompt_toks:
            _bump_stats(prompt=prompt_toks - prefill_seen)
        return job

    # Embedding images is a GPU forward, so it must not overlap generation: the
    # lock now covers the whole prompt build as well. It is taken once, outside
    # run_once, so the forced-tool retry re-generates without re-embedding.
    with gen_lock:
        if images:
            embeddings = embed_images(tokenizer, images)
        input_ids = build_prompt_ids(tokenizer, messages, tools, embeddings)
        prompt_toks = int(input_ids.shape[-1])
        # Default (None) = room left in the context, capped at DEFAULT_MAX_TOKENS
        # so a looping reply can't hold the single slot until the cache fills.
        # Explicit limits are honoured up to the room left.
        room = max(1, int(stats.get("context_length") or 0) - prompt_toks)
        max_tokens = min(room, DEFAULT_MAX_TOKENS) if not max_tokens else min(int(max_tokens), room)
        job = run_once()
        # Forced tool_choice is a prompt nudge; at temperature > 0 the model can
        # occasionally skip the call. One greedy retry makes it deterministic.
        if forced_choice and not parse_tool_calls(text, schemas)[1]:
            temperature = 0.0
            job = run_once()
    seq = job.sequences[0]
    out_toks = int(seq.sequence_ids.seq_len - prompt_toks)
    content, calls = parse_tool_calls(text, schemas)
    if calls:
        finish = "tool_calls"
    else:
        finish = {"max_new_tokens": "length", "eos": "stop",
                  "stop_condition": "stop", "banned": "content_filter"}.get(
                      reason, "stop")
    reasoning, content = split_reasoning(content)
    return text, calls, finish, prompt_toks, out_toks, reasoning, content


async def models(request):
    ctx = stats.get("context_length")
    return web.json_response({"object": "list", "data": [{
        "id": "qwen3.8-27b-exl3-3.5bpw-wm",
        "object": "model",
        "owned_by": "exl3",
        **({"max_model_len": ctx} if ctx else {}),
    }]})


async def health(request):
    with stats_lock:
        return web.json_response({
            "ok": True,
            "busy": gen_lock.locked(),
            "backend": "exl3",
            "vision": bool(vision_state["enabled"]),
            "prompt_tokens_total": stats["prompt_tokens_total"],
            "completion_tokens_total": stats["completion_tokens_total"],
            "context_length": stats["context_length"],
        })


def parse_request(body):
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        return None, "`messages` (list) is required"
    # No limit sent (many agent clients never send one) = use the remaining
    # context, like llama.cpp. A fixed 1024 default cut thinking-mode replies off mid-reasoning.
    _mt = body.get("max_tokens") or body.get("max_completion_tokens")
    max_tokens = int(_mt) if _mt else None
    temperature = float(body.get("temperature", 0.6))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    seed = body.get("seed")
    tools = body.get("tools") or None
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = None
    return dict(
        messages = normalize_messages(messages),
        max_tokens = max_tokens, temperature = temperature,
        top_p = top_p, top_k = top_k,
        seed = int(seed) if seed is not None else None,
        tools = tools,
        tool_choice = body.get("tool_choice"),
        stop = stop,
        stream = bool(body.get("stream", False)),
        model_id = body.get("model", "qwen3.8-27b-exl3-3.5bpw-wm"),
    ), None


async def chat_completions(request):
    app = request.app
    generator, tokenizer = app["generator"], app["tokenizer"]
    try:
        body = await request.json()
    except web.HTTPRequestEntityTooLarge:
        # aiohttp enforces client_max_size inside request.json(); without this
        # branch it falls into the generic handler below and gets misreported
        # as "invalid JSON" (400) even though the body parsed fine.
        return web.json_response(
            {"error": {"message": f"request body exceeds {request.app['max_body_mb']} MiB limit",
                       "type": "invalid_request_error",
                       "code": "request_entity_too_large"}},
            status = 413)
    except Exception:
        return web.json_response({"error": {"message": "invalid JSON"}}, status = 400)
    req, err = parse_request(body)
    if err:
        return web.json_response({"error": {"message": err}}, status = 400)

    import asyncio
    # Images are fetched and decoded before either branch below, so a bad image
    # is a clean 400 rather than an error inside an already-started SSE stream.
    try:
        images = await extract_images(req["messages"])
    except ImageError as e:
        return web.json_response(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status = 400)

    if not req["stream"]:
        try:
            text, calls, finish, ptoks, otoks, reasoning, content = await asyncio.to_thread(
                generate_full, generator, tokenizer, req["messages"],
                req["max_tokens"], req["temperature"], req["top_p"], req["top_k"],
                req["seed"], req["tools"], req["tool_choice"], req["stop"],
                None, images)
        except ImageError as e:
            return web.json_response(
                {"error": {"message": str(e), "type": "invalid_request_error"}},
                status = 400)
        except AssertionError as e:
            return web.json_response(
                {"error": {"message": f"context/cache: {e}", "type": "invalid_request_error"}},
                status = 400)
        msg = {"role": "assistant", "content": content or None}
        if reasoning:
            msg["reasoning_content"] = reasoning
        if calls:
            msg["tool_calls"] = calls
        return web.json_response({
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion", "created": int(time.time()),
            "model": req["model_id"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": otoks,
                      "total_tokens": ptoks + otoks},
        })

    # ---- streaming (SSE) ----
    resp = web.StreamResponse(headers = {
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive"})
    await resp.prepare(request)
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model_id = req["model_id"]
    req_schemas = build_tool_schemas(req["tools"])

    async def run():
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()

        def on_text(chunk):
            loop.call_soon_threadsafe(queue.put_nowait, ("delta", chunk))

        forced_choice = req["tool_choice"] not in (None, "auto", "none")

        def worker():
            try:
                text, calls, finish, ptoks, otoks, reasoning, content = generate_full(
                    generator, tokenizer, req["messages"], req["max_tokens"],
                    req["temperature"], req["top_p"], req["top_k"],
                    req["seed"], req["tools"], req["tool_choice"], req["stop"],
                    on_text = None if forced_choice else on_text,
                    images = images)
                loop.call_soon_threadsafe(queue.put_nowait,
                                          ("done", (calls, finish, reasoning, content)))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        loop.run_in_executor(None, worker)

        async def send(delta, finish = None):
            obj = {"id": cid, "object": "chat.completion.chunk",
                   "created": int(time.time()), "model": model_id,
                   "choices": [{"index": 0, "delta": delta,
                                "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(obj)}\n\n".encode())

        pending, finish, calls_emitted = "", None, False
        call_idx = [0]
        in_think = [True]          # generation starts inside <think> (template)
        THINK_CLOSE = "</think>"

        async def send_call(c):
            nonlocal calls_emitted
            calls_emitted = True
            await send({"tool_calls": [dict(c, index = call_idx[0])]})
            call_idx[0] += 1

        async def flush_pending(final = False):
            """Emit everything parseable from pending; keep marker-safe tail."""
            nonlocal pending
            while True:
                if in_think[0]:
                    close = pending.find(THINK_CLOSE)
                    if close >= 0:
                        head, pending = pending[:close], pending[close + len(THINK_CLOSE):]
                        if head.strip():
                            await send({"reasoning_content": head.lstrip("\n")})
                        in_think[0] = False
                        continue
                    cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                    piece = pending[:cut]
                    if piece.strip():
                        await send({"reasoning_content": piece})
                    pending = pending[cut:]
                    return
                if TOOL_CALL_OPEN in pending:
                    head, rest = pending.split(TOOL_CALL_OPEN, 1)
                    if head.strip() or (final and head):
                        await send({"content": head})
                    if TOOL_CALL_CLOSE in rest:
                        block, pending = rest.split(TOOL_CALL_CLOSE, 1)
                        _, calls = parse_tool_calls(
                            TOOL_CALL_OPEN + block + TOOL_CALL_CLOSE,
                            req_schemas)
                        for c in calls:
                            await send_call(c)
                        continue
                    # unterminated call: final -> implicit close, else hold
                    if final and "<function=" in rest:
                        _, calls = parse_tool_calls(TOOL_CALL_OPEN + rest,
                                                    req_schemas)
                        for c in calls:
                            await send_call(c)
                        pending = ""
                    else:
                        pending = TOOL_CALL_OPEN + rest
                    return
                cut = len(pending) if final else max(0, len(pending) - HOLD_BACK)
                await send({"content": pending[:cut]})
                pending = pending[cut:]
                return

        while True:
            kind, payload = await queue.get()
            if kind == "error":
                await resp.write(
                    f'data: {json.dumps({"error": {"message": payload}})}\n\n'.encode())
                break
            if kind == "delta":
                pending += payload
                await flush_pending()
            elif kind == "done":
                calls, finish, reasoning, content = payload
                await flush_pending(final = True)
                if forced_choice:
                    # Buffered path (no deltas were streamed): emit the
                    # authoritative complete result as deltas.
                    if reasoning:
                        await send({"reasoning_content": reasoning})
                    if content:
                        await send({"content": content})
                if not calls_emitted and calls:
                    for c in calls:
                        await send_call(c)
                await send({}, finish = finish)
                await resp.write(b"data: [DONE]\n\n")
                break
        await resp.write_eof()
    try:
        await run()
    except ConnectionResetError:
        pass
    return resp


def main():
    global MODEL_DIR, DRAFT_DIR, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default = MODEL_DIR)
    ap.add_argument("-dm", "--draft_model", default = DRAFT_DIR,
                    help = "Draft model path, 'mtp' for MTP drafting (head inside the "
                           "main checkpoint: no extra weights, much smaller KV footprint) "
                           "or 'none' to disable drafting")
    ap.add_argument("-gs", "--grid_size", type = int, default = 110)
    ap.add_argument("-cs", "--cache_size", type = int, default = 65536,
                    help = "KV cache size in tokens (default 65536; 8192 default "
                           "of model_init is too small for large tool sets)")
    ap.add_argument("-cq", "--cache_quant", type = str, default = None,
                    help = "Quantized KV cache bits, e.g. 8 or 8,4 (k_bits[,v_bits])")
    ap.add_argument("-p", "--port", type = int, default = PORT)
    ap.add_argument("--host", type = str, default = "0.0.0.0",
                    help = "Interface to bind (use 127.0.0.1 for local-only)")
    ap.add_argument("-ccs", "--cpu_cache_size", type = float, default = 0.0,
                    help = "CPU second-tier cache size in GB (pages spill from "
                           "GPU when the GPU cache is full)")
    ap.add_argument("--vision", action = "store_true",
                    help = "load the checkpoint's vision component and accept "
                           "image parts in chat messages (costs VRAM: 0.86 GiB "
                           "for Qwen3.8-27B, plus each image's context tokens)")
    # Dashed spellings are what start.sh passes; the underscore aliases keep the
    # flag style of the rest of this parser.
    ap.add_argument("--max-image-mb", "--max_image_mb", dest = "max_image_mb",
                    type = float, default = 20.0,
                    help = "max decoded size of a single input image, in MiB")
    ap.add_argument("--max-images", "--max_images", dest = "max_images",
                    type = int, default = 8,
                    help = "max images accepted in one request")
    ap.add_argument("--image-timeout", "--image_timeout", dest = "image_timeout",
                    type = float, default = 15.0,
                    help = "timeout in seconds for fetching one image URL")
    ap.add_argument("--max_body_mb", type = int, default = 64,
                    help = "max request body size in MiB (aiohttp's built-in "
                           "default is 1 MiB, far too small for a full tool "
                           "set + a long transcript)")
    args = ap.parse_args()
    _draft = args.draft_model.lower()
    use_mtp = _draft == "mtp"
    use_draft = _draft not in ("none", "", "-")
    argv = ["-m", args.model,
            "-gs", str(args.grid_size), "-cs", str(args.cache_size)]
    if use_mtp:
        argv += ["-mtp"]
    elif use_draft:
        argv += ["-dm", args.draft_model]
    if args.cache_quant:
        argv += ["-cq", args.cache_quant]
    if args.cpu_cache_size:
        argv += ["-ccs", str(args.cpu_cache_size)]

    print(f" == loading {args.model}"
          + (" + MTP head" if use_mtp else
             (f" + draft {args.draft_model}" if use_draft else " (no draft)"))
          + (" + vision" if args.vision else "")
          + " ...", flush = True)
    generator, tokenizer, vision_model, image_token_id, vision_span_ids = build_model(
        argv, use_draft = use_draft, use_vision = args.vision)
    if vision_model is not None:
        if not 0 <= image_token_id < len(tokenizer.id_to_piece):
            raise SystemExit(f" !! --vision: image_token_id {image_token_id} is "
                             "outside the tokenizer vocabulary")
        vision_state.update({
            "enabled": True,
            "model": vision_model,
            "image_token_id": image_token_id,
            "placeholder": tokenizer.id_to_piece[image_token_id],
            "span": _vision_span(tokenizer, image_token_id, vision_span_ids),
            "max_images": args.max_images,
            "max_image_mb": args.max_image_mb,
            "fetch_timeout": args.image_timeout,
        })
        print(f" == vision enabled (placeholder "
              f"{vision_state['placeholder']!r}, max {args.max_images} image(s) "
              f"per request, {args.max_image_mb} MiB each)", flush = True)
    stats["context_length"] = int(args.cache_size)
    print(" == model ready; accepting requests", flush = True)

    app = web.Application(client_max_size = args.max_body_mb * 1024 * 1024)
    app["generator"] = generator
    app["tokenizer"] = tokenizer
    app["max_body_mb"] = args.max_body_mb
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    web.run_app(app, host = args.host, port = args.port, print = None)


if __name__ == "__main__":
    main()
