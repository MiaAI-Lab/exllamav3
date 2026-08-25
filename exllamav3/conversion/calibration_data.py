import torch
import os
import json
import random


def load_calibration_trace(path, rows, columns, tokenizer):
    """
    Load calibration data from a trace file instead of the bundled default corpus.

    Supported formats:
      * Calibration-trace JSON as published with turboderp's EXL3 releases, e.g.
        turboderp/Qwen3.8-27B-exl3 :: cal_trace.json:
            {"rows": [{"input_ids": [...], "response_ids": [...], ...}, ...], ...}
        Each row's full sequence (input + response) is concatenated, and the combined
        token stream is sliced into fixed-width calibration rows.
      * Plain JSON list of token-ID lists: [[...], [...]] (sequences are concatenated
        and sliced the same way).

    :param path:
        Path to the trace file
    :param rows:
        Number of calibration rows to produce
    :param columns:
        Row length in tokens (cal_cols)
    :param tokenizer:
        Tokenizer (used to validate the trace against the model's vocabulary)
    :return:
        List of (1, columns) token tensors, like get_default_calibration()
    """
    with open(path, "r") as f:
        trace = json.load(f)

    if isinstance(trace, dict) and "rows" in trace:
        seqs = [r["input_ids"] + r.get("response_ids", []) for r in trace["rows"]]
        meta = trace.get("meta", {})
        print(f" -- Calibration trace: {path}")
        print(f"    {len(seqs)} conversations, "
              f"{meta.get('input_tokens', sum(len(r['input_ids']) for r in trace['rows']))} input / "
              f"{meta.get('output_tokens', sum(len(r.get('response_ids', [])) for r in trace['rows']))} output tokens"
              + (f", template vars: {trace['template_vars']}" if trace.get("template_vars") else ""))
    elif isinstance(trace, list):
        seqs = [list(s) for s in trace]
        print(f" -- Calibration trace: {path} ({len(seqs)} sequences)")
    else:
        raise ValueError(f"Unsupported calibration trace format in {path} "
                         f"(expected dict with 'rows' or a list of token-ID lists)")

    vocab_size = tokenizer.actual_vocab_size
    all_tokens = [t for seq in seqs for t in seq]
    max_token = max(all_tokens) if all_tokens else 0
    if max_token >= vocab_size:
        raise ValueError(f"Calibration trace contains token ID {max_token} >= vocab size {vocab_size}; "
                         f"the trace does not match this model's tokenizer")

    t_rows = []
    stream = torch.tensor(all_tokens, dtype = torch.long)
    for i in range(rows):
        a = i * columns
        b = a + columns
        if b > stream.shape[-1]:
            raise ValueError(f"Calibration trace too short: need {rows} x {columns} = {rows * columns} tokens, "
                             f"have {stream.shape[-1]}")
        t_rows.append(stream[a:b].unsqueeze(0))
    print(f"    Using {len(t_rows)} rows x {columns} columns ({len(t_rows) * columns} tokens)")
    return t_rows


def get_calibration(args, tokenizer):
    """
    Return the calibration data for a conversion job: either an external trace
    (args["cal_data"]) or the default bundled corpus.

    :param args:
        Conversion args dict (cal_data, cal_rows, cal_cols)
    :param tokenizer:
        Tokenizer
    :return:
        List of (1, cal_cols) token tensors
    """
    cal_data = args.get("cal_data")
    if cal_data:
        return load_calibration_trace(cal_data, args["cal_rows"], args["cal_cols"], tokenizer)
    return get_default_calibration(args, tokenizer)

def split_art(articles, rows, columns, tokenizer):
    t_rows = []
    idx = 0
    empty = torch.empty((1, 0), dtype = torch.long)
    t_row = empty
    while len(t_rows) < rows:
        add_special_tokens = (len(t_rows) % 2 == 0)
        t_art = tokenizer.encode(articles[idx], add_bos = add_special_tokens, add_eos = add_special_tokens)
        t_row = torch.cat((t_row, t_art), dim = -1)
        t_row = t_row[:, :columns]
        if t_row.shape[-1] == columns:
            t_rows.append(t_row)
            t_row = empty
        idx += 1
    return t_rows


def split_wiki(text, rows, columns, tokenizer):
    articles = [a[a.find("\n") + 1:] for a in text.split("</doc>\n")]
    articles = [a for a in articles if len(a) > 50]
    return split_art(articles, rows, columns, tokenizer)


def split_tiny(text, rows, columns, tokenizer):
    articles = [a.strip() for a in text.split("<|endoftext|>")]
    return split_art(articles, rows, columns, tokenizer)


def shuffle_lines(text, rows, columns, tokenizer):
    articles = text.split("\n")
    articles = [a for a in articles if not a.isspace()]
    random.seed(0)
    random.shuffle(articles)
    return split_art(articles, rows, columns, tokenizer)


def split_raw(text, rows, columns, tokenizer):
    t_all = tokenizer.encode(text)
    t_rows = []
    for i in range(rows):
        a = i * columns
        b = a + columns
        t_rows.append(t_all[:, a:b])
    return t_rows


def random_data(text, rows, columns, tokenizer):
    vocab_size = tokenizer.actual_vocab_size
    torch.manual_seed(0)
    t_rows = []
    for i in range(rows):
        t_row = torch.randint(0, vocab_size, (1, columns), dtype = torch.long)
        t_rows.append(t_row)
    return t_rows


def get_default_calibration(args, tokenizer):
    """
    Return the default calibration corpus as a list of token rows.

    The final calibration data is a weighted mix of fixed-width token slices drawn from several bundled text
    sources, plus random token rows to exercise the full vocabulary. Each row is a 1 x cal_cols tensor, and the
    number of rows is distributed across sources according to their configured weights.
    """
    columns = args["cal_cols"]
    rows = args["cal_rows"]

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "standard_cal_data")
    files = [
        ("c4.utf8", 20, shuffle_lines),
        ("code.utf8", 20, split_raw),
        ("multilingual.utf8", 10, shuffle_lines),
        ("technical.utf8", 10, split_raw),
        ("wiki.utf8", 50, split_wiki),
        ("tiny.utf8", 5, split_tiny),
        (None, 20, random_data),
    ]

    dist_sum = sum(x for (_, x, _) in files)
    cal_data = []

    for filename, weight, processor in files:
        target_rows = max(1, int(weight / dist_sum * rows))
        if filename:
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding = "utf8") as f:
                file_text = f.read()
        else:
            file_text = None
            target_rows = max(1, rows - len(cal_data))
        r = processor(file_text, target_rows, columns, tokenizer)
        cal_data += r

    # cal_data = torch.cat(cal_data, dim = 0)
    return cal_data
