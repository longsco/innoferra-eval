#!/usr/bin/env python3
"""Synthetic multi-turn driver for the draft-window smoke (innoferra 10-06, next180 serving track). No customer data: every
prompt is generated from a fixed word list with a seeded RNG.

Each session = one long synthetic document (shared by --shared-docs groups, so sessions share long prefixes) + a session tail,
then --turns turns: turn k+1 sends the previous prompt + the model's previous output + a short new user line, so every turn
hits the radix cache on the previous sequence (the agentic pattern), with HiCache pressure once sessions x context exceeds the
device pool. Requests go to the engine's native /generate (temperature 0) so meta_info carries cached_tokens and the DSpark
counters (spec_verify_ct, spec_accept_length).
Output: one JSON line per request (session, turn, prompt_tokens, cached_tokens, completion_tokens, spec_verify_ct,
spec_accept_length, latency, sha1 of the output text). Text itself is not stored.
usage: mt_driver.py --url http://127.0.0.1:19491 --sessions 24 --turns 5 --concurrency 1 --doc-words 20000 --out x.jsonl
"""
import argparse
import hashlib
import json
import random
import threading
import time
import urllib.request

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi "
         "psi omega river stone cloud field signal vector matrix kernel cache window draft target verify page token buffer "
         "stream graph module server request session context memory planner window budget parity restore release pin").split()


def text(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def post(url, payload, timeout):
    req = urllib.request.Request(url + "/generate", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_session(a, sid, out, lock):
    rng = random.Random(a.seed * 1000003 + sid)
    doc_rng = random.Random(a.seed * 7919 + (sid % max(1, a.shared_docs)))
    prompt = "Document:\n" + text(doc_rng, a.doc_words) + "\nSession %d notes: %s\n" % (sid, text(rng, a.tail_words))
    for turn in range(a.turns):
        prompt += "User: %s\nAssistant:" % text(rng, rng.randint(10, a.user_words))
        t0 = time.time()
        try:
            r = post(a.url, {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": a.max_new,
                                                                  "ignore_eos": False}}, a.timeout)
        except Exception as e:  # noqa: BLE001
            with lock:
                out.write(json.dumps(dict(session=sid, turn=turn, error=f"{type(e).__name__}: {str(e)[:200]}")) + "\n")
            return
        mi = r.get("meta_info", {})
        o = r.get("text", "")
        rec = dict(session=sid, turn=turn, prompt_tokens=mi.get("prompt_tokens"), cached_tokens=mi.get("cached_tokens"),
                   completion_tokens=mi.get("completion_tokens"), spec_verify_ct=mi.get("spec_verify_ct"),
                   spec_accept_length=mi.get("spec_accept_length"), latency=round(time.time() - t0, 3),
                   out_sha1=hashlib.sha1(o.encode()).hexdigest())
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
        prompt += o + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--sessions", type=int, default=24)
    p.add_argument("--turns", type=int, default=5)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--doc-words", type=int, default=20000)
    p.add_argument("--tail-words", type=int, default=300)
    p.add_argument("--user-words", type=int, default=80)
    p.add_argument("--shared-docs", type=int, default=4)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lock = threading.Lock()
    with open(a.out, "w") as out:
        sids = list(range(a.sessions))
        if a.concurrency <= 1:
            for s in sids:
                run_session(a, s, out, lock)
        else:
            it = iter(sids)
            it_lock = threading.Lock()

            def worker():
                while True:
                    with it_lock:
                        s = next(it, None)
                    if s is None:
                        return
                    run_session(a, s, out, lock)

            th = [threading.Thread(target=worker) for _ in range(a.concurrency)]
            for t in th:
                t.start()
            for t in th:
                t.join()
    print("done", a.out)


if __name__ == "__main__":
    main()
