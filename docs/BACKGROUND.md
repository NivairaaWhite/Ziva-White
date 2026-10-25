<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
The supervised background daemon.

# Running in the background

```bash
python -m zivawhite.app run                # foreground: web UI + API + jobs (Ctrl+C stops)
python -m zivawhite.app daemon start       # the same, detached; no terminal needed
python -m zivawhite.app daemon status      # state, each job, model, learning progress
python -m zivawhite.app daemon logs -n 100
python -m zivawhite.app daemon stop        # clean shutdown; force-kills the process tree only if it hangs
python -m zivawhite.app daemon install     # start at login: systemd user service / LaunchAgent / Task Scheduler
python -m zivawhite.app daemon uninstall
python -m zivawhite.app run --no-server    # jobs only, no web server (e.g. a voice-only box)
```

Jobs (`background.jobs`): **autolearn** (the datasets watcher), **timers**, and **voice** (only if
`voice.enabled`). Each runs supervised: a crash restarts it with exponential backoff, repeated crashes
mark it `failed` instead of looping forever, and one that can't work on this machine (no microphone, no
speech model) is marked `unavailable` with the reason and not retried. State goes to
`runs/daemon_status.json` every few seconds; logs to `runs/daemon.log` with size rotation. Timers set
through `command` fire only while something is running.
