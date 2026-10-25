"""
The single entrypoint.

    python -m zivawhite.app datasets               # what is in the datasets folder, what still needs training
    python -m zivawhite.app tokenizer --vocab_size 8000   # (optional) learn a vocabulary from YOUR data, before first training
    python -m zivawhite.app autolearn              # watch datasets/ and train on whatever you drop there
    python -m zivawhite.app autolearn --once       # one scan/train round, then exit
    python -m zivawhite.app run                    # web UI + API + background jobs (autolearn, voice, timers)
    python -m zivawhite.app daemon start           # the same, detached, no terminal needed
    python -m zivawhite.app daemon install         # ...and start it at login
    python -m zivawhite.app voice                  # voice assistant in this terminal (wake word, mic, speaker)
    python -m zivawhite.app voice --test-mic       # see what the recogniser hears; tune wake words
    python -m zivawhite.app command "volume up"    # one command or question, same path as voice
    python -m zivawhite.app ask "question"         # one-shot routed answer
    python -m zivawhite.app serve                  # just the HTTP API/web UI (no background jobs)
    python -m zivawhite.app inspect                # which checkpoint would be loaded, and what is it
    python -m zivawhite.app train ...              # train/continue-train directly (see train.py)
    python -m zivawhite.app generate --prompt ..   # raw local-model sampling only
    python -m zivawhite.app ingest --commit        # fold the web self-training queue into shards (legacy path)

There is no external AI-API wrapper anywhere: every answer is written by this project's own
trained weights. Everything configurable lives in configs/ziva.yaml.
"""
from __future__ import annotations

import argparse
import json
import atexit
import os
import signal
import sys
import threading
from typing import List, Optional

from . import __version__
from .config import Config, get_config


# ----------------------------------------------------------------------------- small helpers
def _cfg() -> Config:
    cfg = get_config()
    for problem in cfg.problems():
        print(f"[config] warning: {problem}", file=sys.stderr)
    return cfg


def _text_shards(cfg: Config) -> str:
    return str(cfg.data_dir / "datasets" / "text" / "shards")


# ------------------------------------------------------------------- retained model commands
def cmd_train(args: argparse.Namespace) -> None:
    from . import train as train_mod

    sys.argv = ["train.py"] + args.rest
    train_mod.main()


def cmd_compress(args: argparse.Namespace) -> None:
    from . import compress as compress_mod

    sys.argv = ["compress.py", "--checkpoint", args.checkpoint]
    if args.out:
        sys.argv += ["--out", args.out]
    if args.data_dir:
        sys.argv += ["--data_dir", args.data_dir]
    compress_mod.main()


def cmd_generate(args: argparse.Namespace) -> None:
    from . import infer as infer_mod

    sys.argv = ["infer.py", "--prompt", args.prompt, "--max_new_tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature)]
    if args.checkpoint:
        sys.argv += ["--checkpoint", args.checkpoint]
    if args.raw:
        sys.argv.append("--raw")
    infer_mod.main()


def cmd_ask(args: argparse.Namespace) -> None:
    from .runtime import Runtime

    rt = Runtime(_cfg(), args.checkpoint)
    known = rt.assistant.identity.answer(args.query)
    if known is not None:
        print("[route=identity | reason=answered from the assistant's configuration | 0.00s]")
        print(known)
        return
    if rt.router is None:
        print(rt.router_error)
        sys.exit(1)
    answer = rt.router.ask(args.query, route=args.route)
    print(f"[route={answer.route} | reason={answer.reason} | {answer.latency_s:.2f}s]")
    if answer.sources:
        print(f"[sources: {', '.join(answer.sources)}]")
    print(answer.text)


def cmd_agent(args: argparse.Namespace) -> None:
    """Answer with tools: the model may call the calculator / clock (and, with --web, web search/fetch) in a bounded
    loop. Read-only tools only; device control is not reachable from here (see tools.py)."""
    from .runtime import Runtime
    from .tools import TOOL_CALL_CLOSE, default_registry, run_agent

    rt = Runtime(_cfg(), args.checkpoint)
    if rt.router is None or not rt.router.has_local_model():
        print(rt.router_error or "no local model yet (train one, or pass --checkpoint)")
        sys.exit(1)
    if not rt.router.behavior_tuned:
        print("[warning] this checkpoint is not chat-tuned, so it will not follow the tool protocol. Run `sft` first "
              "(`tool-data` writes training conversations for it).")
    registry = default_registry(rt.router.connectors, web=args.web)
    chat_fn = lambda msgs: rt.router.chat(msgs, max_new_tokens=args.max_new_tokens, stop=[TOOL_CALL_CLOSE])["text"]  # noqa: E731
    result = run_agent(chat_fn, args.query, registry, max_steps=args.max_steps)
    for i, step in enumerate(result.steps, 1):
        print(f"[tool {i}] {step['tool']}({json.dumps(step['arguments'], ensure_ascii=False)}) -> {step['result']}")
    print(result.text or "(no answer)")


def cmd_tool_data(args: argparse.Namespace) -> None:
    from .tools import write_tool_conversations

    n = write_tool_conversations(args.out, args.n, args.seed)
    print(f"wrote {n} verifiable tool-use conversations -> {args.out}\n"
          f"mix them into your SFT data (about 5-10%): python -m zivawhite.app sft --data {args.out} <your other data> ...")


def cmd_inspect(args: argparse.Namespace) -> None:
    from .checkpoint import describe_checkpoint, find_default_checkpoint, resolve_serving_checkpoint
    from .device import resolve_runtime
    from .runtime import Runtime

    cfg = _cfg()
    path = args.checkpoint or find_default_checkpoint(str(cfg.root), run_dirs=Runtime(cfg, load_model=False).run_dirs())
    if not path:
        print(f"no checkpoint found yet. Put training files in {cfg.datasets_dir} and run: "
              f"python -m zivawhite.app autolearn --once")
        return
    served = resolve_serving_checkpoint(path)
    print(f"asked for : {path}\nwould load: {served}")
    for key, value in describe_checkpoint(served).items():
        print(f"  {key:15}: {value}")
    print(f"  this machine   : {resolve_runtime().summary()}")


def cmd_ingest(args: argparse.Namespace) -> None:
    from .data.ingest import fold_queue

    cfg = _cfg()
    queue = args.queue_dir or str(cfg.data_dir / "queue")
    corpus = args.corpus_out or str(cfg.data_dir / "corpus" / "web_ingest.txt")
    manifest = args.manifest_out or str(cfg.data_dir / "corpus" / "manifest.jsonl")
    shards = args.shards_out or str(cfg.data_dir / "processed_web")
    n = fold_queue(queue, corpus, manifest)
    print(f"[zivawhite] folded {n} queued documents into {corpus}")
    if args.commit and n > 0:
        from .autolearn import model_tokenizer_dir
        from .data.ingest import build_shards

        tokenizer_dir = args.tokenizer_dir or model_tokenizer_dir(cfg)     # the model's own vocabulary, not the bundled default
        if tokenizer_dir:
            tokenizer_dir = str(cfg.resolve(tokenizer_dir))
        n_train, n_val = build_shards(corpus, tokenizer_dir, shards)
        print(f"[zivawhite] rebuilt shards: {n_train:,} train tokens, {n_val:,} val tokens -> {shards}")


def cmd_nas(args: argparse.Namespace) -> None:
    from .autolearn import model_tokenizer_dir
    from .device import resolve_runtime
    from .nas import report, search
    from .tokenizer import TokenizerNotFound, load_tokenizer

    rt = resolve_runtime()
    cfg = _cfg()
    tokenizer_dir = args.tokenizer_dir or model_tokenizer_dir(cfg)             # the model's own vocabulary
    try:
        tok = load_tokenizer(str(cfg.resolve(tokenizer_dir)) if tokenizer_dir else None)
    except TokenizerNotFound as e:
        raise SystemExit(f"[nas] {e}") from None
    results = search(data_dir=args.data_dir or _text_shards(cfg), vocab_size=tok.vocab_size, block_size=args.block_size,
                     device=rt.device, max_params=args.max_params, trials=args.trials,
                     search_steps=args.search_steps, batch_size=args.batch_size, seed=args.seed)
    print()
    print(report(results, out_path=args.out))
    print(f"\nwinner: n_layer={results[0].n_layer} n_head={results[0].n_head} n_embd={results[0].n_embd}")
    print("this is a report only -- nothing was deployed. Train the winner yourself if you like it.")


def cmd_consolidate(args: argparse.Namespace) -> None:
    from .checkpoint import load_checkpoint
    from .continual import compute_fisher, save_consolidation
    from .device import resolve_runtime
    from .model import ModelConfig, ZivaGPT

    rt = resolve_runtime()
    ckpt = load_checkpoint(args.checkpoint, map_location=rt.device)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = ZivaGPT(cfg).to(rt.device)
    model.load_state_dict(ckpt["model_state"])
    data_dir = args.data_dir or _text_shards(_cfg())
    fisher = compute_fisher(model, data_dir, cfg.block_size, args.batch_size, rt.device, args.n_batches)
    save_consolidation(args.out, model, fisher)
    print(f"[zivawhite] consolidated task knowledge from {data_dir} -> {args.out}")
    print(f"[zivawhite] use it by setting autolearn.ewc: {args.out} in configs/ziva.local.yaml, or pass --ewc to train")


def cmd_tune(args: argparse.Namespace) -> None:
    from .device import benchmark_and_tune, resolve_runtime
    from .model import ModelConfig

    rt = resolve_runtime()
    print(f"[zivawhite] {rt.summary()}")
    results = benchmark_and_tune(rt, lambda tier: ModelConfig.preset(tier, vocab_size=32000, block_size=args.block_size))
    for bs, tps in sorted(results.items()):
        flag = "  <- fastest measured" if tps == max(results.values()) else ""
        print(f"  batch_size={bs:>4}: {tps:>10,.0f} tok/s{flag}")
    best = max(results, key=results.get)
    print(f"\nempirically best batch_size on this machine: {best}  (add \"--batch_size\", \"{best}\" to autolearn.train_args)")


# ------------------------------------------------------------------------------- learning
def cmd_autolearn(args: argparse.Namespace) -> None:
    from . import autolearn as autolearn_mod

    argv = []
    for flag in ("once", "status", "no_throttle"):
        if getattr(args, flag):
            argv.append("--" + flag.replace("_", "-"))
    autolearn_mod.main(argv)


def cmd_tokenizer(args: argparse.Namespace) -> None:
    """Train a tokenizer on the datasets folder (optional: the bundled general-English one is the default)."""
    from .checkpoint import find_default_checkpoint
    from .data.folder import DatasetFolder
    from .data.tokenizer_train import NotEnoughText, default_out_dir, train_from_datasets
    from .runtime import Runtime
    from .tokenizer import BUNDLED_TOKENIZER_DIR, has_tokenizer

    cfg = _cfg()
    folder = DatasetFolder.from_config(cfg)
    folder.ensure_layout()
    out = cfg.resolve(args.out) if args.out else default_out_dir(cfg)
    if has_tokenizer(str(out)) and not args.force:
        raise SystemExit(f"[tokenizer] {out} already holds a tokenizer. Choose another --out, or pass --force to replace it "
                         f"(only safe if no model was trained with it).")
    if os.path.realpath(str(out)) == os.path.realpath(BUNDLED_TOKENIZER_DIR):
        raise SystemExit("[tokenizer] refusing to overwrite the bundled tokenizer: choose another --out.")
    existing = find_default_checkpoint(str(cfg.root), run_dirs=Runtime(cfg, load_model=False).run_dirs())
    try:
        result = train_from_datasets(folder, args.vocab_size, str(out), force=args.force)
    except NotEnoughText as e:
        raise SystemExit(f"[tokenizer] {e}") from None
    try:
        shown = out.relative_to(cfg.root).as_posix()
    except ValueError:
        shown = str(out)
    print(f"[tokenizer] trained a {result['vocab_size']:,}-token vocabulary on your datasets folder -> {out}")
    print("\nTo use it, put this in configs/ziva.local.yaml, BEFORE the first training run:\n"
          f"  autolearn:\n    tokenizer_dir: {shown}\n"
          "then run: python -m zivawhite.app autolearn --once")
    if existing:
        print(f"\nNOTE: a model already exists ({existing}). It keeps the vocabulary it was trained with; this new one is for "
              f"a NEW model. To start over with it, move or delete runs/ziva and runs/ziva_behavior first.")


def cmd_datasets(args: argparse.Namespace) -> None:
    from . import autolearn as autolearn_mod
    from .data.folder import DatasetFolder

    cfg = _cfg()
    if args.init:
        DatasetFolder.from_config(cfg).ensure_layout()
        print(f"[datasets] ready: {cfg.datasets_dir}\n  text/   prose to learn language from\n  chat/   conversations to learn answering from\n"
              f"  (anything else is sorted automatically; see {cfg.datasets_dir / 'README.md'})")
    autolearn_mod.print_status(cfg)


# ------------------------------------------------------------------------------ assistant
def cmd_command(args: argparse.Namespace) -> None:
    from .runtime import Runtime

    rt = Runtime(_cfg(), args.checkpoint, load_model=not args.no_model)
    reply = rt.assistant.handle(" ".join(args.text), session="cli")
    print(reply.text)
    if reply.action == "timer_set" and reply.ok:
        print("(timers fire while `run`, `daemon start` or `voice` is running)")
    if reply.kind == "confirm":
        try:
            answer = input("> ")
        except EOFError:
            answer = "no"
        print(rt.assistant.handle(answer, session="cli").text)


def _install_stop_signals(stop: threading.Event) -> None:
    def handler(signum, frame):
        stop.set()
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):      # not the main thread / unsupported on this platform
                pass


def cmd_voice(args: argparse.Namespace) -> None:
    cfg = _cfg()
    from .voice.errors import VoiceUnavailable

    try:
        if args.devices:
            from .voice.audio import list_input_devices

            devices = list_input_devices()
            for d in devices:
                print(f"  [{d['index']}] {d['name']}  ({d['channels']} ch, {d['default_rate']} Hz)")
            print("set voice.stt.input_device in configs/ziva.local.yaml to the index or name you want" if devices else "no input devices found")
            return
        if args.say:
            from .voice.tts import build_tts

            build_tts(cfg).speak(" ".join(args.say))
            return
        if args.test_mic:
            from .voice.loop import test_microphone

            test_microphone(cfg, seconds=args.seconds)
            return
        from .runtime import Runtime

        rt = Runtime(cfg, args.checkpoint, load_model=not args.no_model)
        stop = threading.Event()
        _install_stop_signals(stop)
        print("[voice] press Ctrl+C to stop")
        timers = threading.Thread(target=rt.timers.run, args=(stop,), daemon=True)
        timers.start()
        rt._run_voice(stop)
    except VoiceUnavailable as e:
        print(f"[voice] cannot start: {e}")
        sys.exit(1)


# ------------------------------------------------------------------------- serve / run / daemon
def _uvicorn_server(app, host: str, port: int):
    import uvicorn

    return uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_config=None))


def _warn_exposed(cfg: Config, host: str) -> None:
    from .security import get_server_api_key

    if host not in ("127.0.0.1", "localhost", "::1") and not get_server_api_key():
        print(f"[server] warning: listening on {host} with no ZIVAWHITE_SERVER_API_KEY: anyone on your network can "
              f"use this server (device control stays restricted to this machine)")


def cmd_serve(args: argparse.Namespace) -> None:
    import logging

    from .runtime import Runtime
    from .server import create_app

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _cfg()
    rt = Runtime(cfg, args.checkpoint)
    host, port = args.host or cfg.get("background.host", "127.0.0.1"), args.port or int(cfg.get("background.port", 8000))
    _warn_exposed(cfg, host)
    timers_stop = threading.Event()
    threading.Thread(target=rt.timers.run, args=(timers_stop,), daemon=True).start()   # "set a timer" works from the web UI too
    try:
        _uvicorn_server(create_app(rt), host, port).run()
    finally:
        timers_stop.set()


def cmd_run(args: argparse.Namespace) -> None:
    """Web UI + API + every background job, supervised, until stopped."""
    import logging

    from . import daemon
    from .runtime import Runtime

    cfg = _cfg()
    if args.daemon:
        daemon.setup_daemon_logging(cfg)
        daemon.write_pid(cfg)
        atexit.register(daemon.clear_pid, cfg)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = daemon.daemon_paths(cfg)
    bg = cfg.section("background")
    host, port = args.host or bg.get("host", "127.0.0.1"), args.port or int(bg.get("port", 8000))

    rt = Runtime(cfg, args.checkpoint)
    holder: dict = {}

    def stop_server() -> None:
        if "server" in holder:
            holder["server"].should_exit = True

    sup = daemon.Supervisor(rt.jobs(), max_restarts=int(bg.get("max_restarts", 5)), heartbeat_s=float(bg.get("heartbeat_s", 5)),
                            status_path=paths["status"], stop_file=paths["stop"], extra_status=rt.status,
                            on_stop_request=stop_server)
    names = ", ".join(rt.jobs()) or "none"
    print(f"[run] Ziva White {__version__}; background jobs: {names}")
    sup.start()
    try:
        if args.no_server:
            stop = sup.stop_event
            _install_stop_signals(stop)
            print("[run] no web server (--no-server); Ctrl+C to stop")
            while not stop.is_set():
                stop.wait(1.0)
        else:
            from .server import create_app

            _warn_exposed(cfg, host)
            holder["server"] = _uvicorn_server(create_app(rt, sup.snapshot), host, port)
            print(f"[run] web UI: http://{'localhost' if host in ('0.0.0.0', '::') else host}:{port}")
            holder["server"].run()
    finally:
        sup.stop(timeout=30)
        print("[run] stopped")


def cmd_daemon(args: argparse.Namespace) -> None:
    from . import daemon

    cfg = _cfg()
    action = args.action
    if action == "start":
        ok, msg = daemon.start_daemon(cfg)
        print(f"[daemon] {msg}")
        sys.exit(0 if ok else 1)
    elif action == "stop":
        ok, msg = daemon.stop_daemon(cfg)
        print(f"[daemon] {msg}")
        sys.exit(0 if ok else 1)
    elif action == "restart":
        daemon.stop_daemon(cfg)
        ok, msg = daemon.start_daemon(cfg)
        print(f"[daemon] {msg}")
        sys.exit(0 if ok else 1)
    elif action == "status":
        st = daemon.daemon_status(cfg)
        print(f"state   : {st['state']}" + (f"  (pid {st['pid']}, up {st.get('uptime_s', '?')}s)" if st.get("pid") and st["state"] in ("running", "unresponsive") else ""))
        if st.get("note"):
            print(f"note    : {st['note']}")
        for name, j in (st.get("jobs") or {}).items():
            extra = f"  restarts={j['restarts']}" if j["restarts"] else ""
            err = f"  last error: {j['last_error']}" if j.get("last_error") else ""
            print(f"job     : {name:10} {j['state']}{extra}{err}")
        rt = st.get("runtime") or {}
        if rt.get("model"):
            m = rt["model"]
            print(f"model   : {m.get('file')} (step {m.get('step')})")
        elif st["state"] == "running":
            print(f"model   : none yet ({rt.get('model_error') or 'waiting for training data'})")
        al = rt.get("autolearn") or {}
        if al:
            print(f"learning: {al.get('state')}" + (f"  - {al['waiting']}" if al.get("waiting") else ""))
        print(f"log     : {st['log']}")
        sys.exit(0 if st["state"] == "running" else 3)
    elif action == "logs":
        print(daemon.tail_log(cfg, args.lines) or "(no log yet)")
    elif action == "install":
        ok, msg = daemon.install_autostart(cfg)
        print(f"[daemon] {msg}")
        sys.exit(0 if ok else 1)
    elif action == "uninstall":
        ok, msg = daemon.uninstall_autostart(cfg)
        print(f"[daemon] {msg}")
        sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="zivawhite")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train or continue-train the local model (any device)")
    p_train.set_defaults(func=cmd_train)

    # These hand their whole argument list to the module's own parser (see main()); they are listed here so
    # `--help` shows them. Run e.g. `python -m zivawhite.app sft --help` for each one's flags.
    for name, text in (
        ("sft", "supervised chat fine-tuning: learns only the assistant's replies (system/tool roles, epochs, early stopping)"),
        ("dpo", "Direct Preference Optimization on top of a chat-tuned model (incl. self-play pair building)"),
        ("eval", "perplexity / bits-per-byte on held-out text, assistant-only chat loss, repetition metrics"),
        ("export", "write a Hugging Face LlamaForCausalLM folder (safetensors + tokenizer + model card + Ollama Modelfile); --push uploads it"),
        ("import-hf", "start from an open Llama-architecture model (e.g. SmolLM2): convert it to a ZivaWhite checkpoint"),
        ("plan", "compute planner: how long would this take / what model fits my GPU budget (advice only)"),
        ("data", "download and convert Hugging Face datasets (text shards, chat JSONL, preference pairs, tokenizer)"),
    ):
        sub.add_parser(name, help=text).add_argument("rest", nargs=argparse.REMAINDER)

    p_agent = sub.add_parser("agent", help="answer with tools (calculator, clock, optional web): a bounded, read-only tool loop")
    p_agent.add_argument("query")
    p_agent.add_argument("--checkpoint", default=None)
    p_agent.add_argument("--web", action="store_true", help="also allow web_search / web_fetch (only if configured)")
    p_agent.add_argument("--max_steps", type=int, default=4)
    p_agent.add_argument("--max_new_tokens", type=int, default=256)
    p_agent.set_defaults(func=cmd_agent)

    p_td = sub.add_parser("tool-data", help="write verifiable synthetic tool-use training conversations (no outside data)")
    p_td.add_argument("--out", default="data/chat/tool_use.jsonl")
    p_td.add_argument("--n", type=int, default=2000)
    p_td.add_argument("--seed", type=int, default=1337)
    p_td.set_defaults(func=cmd_tool_data)

    p_gen = sub.add_parser("generate", help="raw local-model sampling, no routing")
    p_gen.add_argument("--checkpoint", default=None, help="default: the newest run configured under autolearn.*_run")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max_new_tokens", type=int, default=200)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--raw", action="store_true", help="plain continuation, no chat template")
    p_gen.set_defaults(func=cmd_generate)

    p_compress = sub.add_parser("compress", help="dynamic int8-quantize a checkpoint for faster/smaller CPU inference")
    p_compress.add_argument("--checkpoint", required=True)
    p_compress.add_argument("--out", default=None, help="default: checkpoint_int8.pt next to --checkpoint")
    p_compress.add_argument("--data_dir", default=None, help="report before/after val perplexity using this shard")
    p_compress.set_defaults(func=cmd_compress)

    p_ask = sub.add_parser("ask", help="one-shot routed answer (local / web, auto-decided) -- no API wrapper")
    p_ask.add_argument("query")
    p_ask.add_argument("--checkpoint", default=None)
    p_ask.add_argument("--route", default="auto", choices=["auto", "local", "web"])
    p_ask.set_defaults(func=cmd_ask)

    p_ingest = sub.add_parser("ingest", help="fold the web self-training queue into a corpus + shards (legacy path; autolearn "
                                             "with autolearn.web.enabled does this into the datasets folder instead)")
    for opt in ("queue_dir", "corpus_out", "manifest_out", "shards_out", "tokenizer_dir"):
        p_ingest.add_argument("--" + opt, default=None)
    p_ingest.add_argument("--commit", action="store_true", help="also rebuild train/val shards")
    p_ingest.set_defaults(func=cmd_ingest)

    p_auto = sub.add_parser("autolearn", help="watch the datasets folder and train on what you put there")
    p_auto.add_argument("--once", action="store_true", help="one scan/train round, then exit")
    p_auto.add_argument("--status", action="store_true", help="show what is in the folder and what still needs training")
    p_auto.add_argument("--no-throttle", dest="no_throttle", action="store_true", help="train even if the machine is busy or on battery")
    p_auto.set_defaults(func=cmd_autolearn)

    p_ds = sub.add_parser("datasets", help="show the datasets folder: files, what changed, what still needs training")
    p_ds.add_argument("--init", action="store_true", help="create the folder layout (text/, chat/, _sources/, _eval/)")
    p_ds.set_defaults(func=cmd_datasets)

    p_tok = sub.add_parser("tokenizer", help="(optional) learn a vocabulary from your datasets folder, instead of the bundled "
                                             "general-English one; do this BEFORE the first training run")
    p_tok.add_argument("--vocab_size", type=int, default=8000, help="tokens in the vocabulary (default 8000; needs ~150 characters of text per token)")
    p_tok.add_argument("--out", default=None, help="where to save it (default: <data>/tokenizer)")
    p_tok.add_argument("--force", action="store_true", help="replace an existing tokenizer folder / train on too little text anyway")
    p_tok.set_defaults(func=cmd_tokenizer)

    p_nas = sub.add_parser("nas", help="bounded architecture search -- produces a report, deploys nothing")
    p_nas.add_argument("--data_dir", default=None, help="default: the text shards autolearn builds from datasets/")
    p_nas.add_argument("--tokenizer_dir", default=None)
    p_nas.add_argument("--block_size", type=int, default=128)
    p_nas.add_argument("--max_params", type=int, default=2_000_000)
    p_nas.add_argument("--trials", type=int, default=6)
    p_nas.add_argument("--search_steps", type=int, default=100)
    p_nas.add_argument("--batch_size", type=int, default=8)
    p_nas.add_argument("--seed", type=int, default=0)
    p_nas.add_argument("--out", default=None, help="optional path to write results as JSON")
    p_nas.set_defaults(func=cmd_nas)

    p_cons = sub.add_parser("consolidate", help="compute Fisher info (EWC) so future training doesn't forget this data")
    p_cons.add_argument("--checkpoint", required=True)
    p_cons.add_argument("--data_dir", default=None, help="default: the text shards autolearn builds from datasets/")
    p_cons.add_argument("--out", default="runs/consolidation.pt")
    p_cons.add_argument("--batch_size", type=int, default=8)
    p_cons.add_argument("--n_batches", type=int, default=50)
    p_cons.set_defaults(func=cmd_consolidate)

    p_tune = sub.add_parser("tune", help="empirically benchmark batch sizes on this machine")
    p_tune.add_argument("--block_size", type=int, default=512)
    p_tune.set_defaults(func=cmd_tune)

    p_inspect = sub.add_parser("inspect", help="show which checkpoint ask/serve would load, and what kind of model it is")
    p_inspect.add_argument("--checkpoint", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    p_cmd = sub.add_parser("command", help="run one command or question through the assistant (same path as voice)")
    p_cmd.add_argument("text", nargs="+")
    p_cmd.add_argument("--checkpoint", default=None)
    p_cmd.add_argument("--no-model", dest="no_model", action="store_true", help="don't load the language model (faster for device commands)")
    p_cmd.set_defaults(func=cmd_command)

    p_voice = sub.add_parser("voice", help="the voice assistant in this terminal: wake word, microphone, speaker")
    p_voice.add_argument("--devices", action="store_true", help="list microphones")
    p_voice.add_argument("--test-mic", dest="test_mic", action="store_true", help="show what the recogniser hears and whether it counts as a wake word; acts on nothing")
    p_voice.add_argument("--seconds", type=float, default=30.0, help="how long --test-mic listens")
    p_voice.add_argument("--say", nargs="+", default=None, metavar="TEXT", help="speak this text and exit (tests the speaker)")
    p_voice.add_argument("--checkpoint", default=None)
    p_voice.add_argument("--no-model", dest="no_model", action="store_true", help="don't load the language model")
    p_voice.set_defaults(func=cmd_voice)

    for name, helptext in (("serve", "just the HTTP API and web UI (no background jobs)"),
                           ("run", "web UI + API + background jobs (autolearn, voice, timers), supervised")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("--checkpoint", default=None)
        sp.add_argument("--host", default=None, help="default: background.host (127.0.0.1 = this machine only). "
                                                      "0.0.0.0 exposes it to your network: set ZIVAWHITE_SERVER_API_KEY first")
        sp.add_argument("--port", type=int, default=None, help="default: background.port")
        sp.set_defaults(func=cmd_serve if name == "serve" else cmd_run)
        if name == "run":
            sp.add_argument("--daemon", action="store_true", help="(used by `daemon start` and autostart) log to the rotated file and write the pid file")
            sp.add_argument("--no-server", dest="no_server", action="store_true", help="background jobs only, no web server")

    p_d = sub.add_parser("daemon", help="run in the background: start / stop / status / logs / install (start at login)")
    p_d.add_argument("action", choices=["start", "stop", "restart", "status", "logs", "install", "uninstall"])
    p_d.add_argument("-n", "--lines", type=int, default=50, help="lines for `logs`")
    p_d.set_defaults(func=cmd_daemon)
    return p


# subcommand -> module whose main(argv) owns that command's flags
_DELEGATED = {"sft": "sft", "dpo": "dpo", "eval": "evaluate", "export": "export_hf", "import-hf": "import_hf",
              "plan": "plan", "data": "data.hf_prepare"}


def main(argv: Optional[List[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    # "train" delegates its entire argv straight to train.py's own parser -- handled here, before
    # the top-level parser sees it, because argparse's REMAINDER doesn't reliably swallow
    # "--flag value" pairs placed right after a subcommand (a known argparse quirk).
    if argv and argv[0] == "train":
        cmd_train(argparse.Namespace(rest=argv[1:]))
        return
    if argv and argv[0] in _DELEGATED:
        import importlib

        importlib.import_module(f"zivawhite.{_DELEGATED[argv[0]]}").main(argv[1:])
        return
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
