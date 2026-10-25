<!-- Moved verbatim out of the top-level README, which had grown past 45 KB. Back to the [README](../README.md). -->
What the assistant may do to this machine, and how it is kept safe.

# Device control

Say it or type it; both take the same path. Matched by a fixed grammar (`control/intents.py`):

| Say | Does |
|---|---|
| volume up / down / "set the volume to 40" / mute / unmute | system volume |
| brightness up / down / "set the brightness to 70" | screen brightness |
| play, pause, next song, previous track | media keys / players |
| take a screenshot | saves a PNG under `device_control.screenshot_dir` |
| open calculator / open the terminal app | only apps listed in `device_control.apps` |
| go to youtube dot com / open website example.com | http(s) only, real hostnames only |
| lock the screen / sleep / restart / shut down | asks "are you sure?" first |
| what time is it / the date / battery / CPU usage | answers directly |
| set a timer for 10 minutes / remind me in an hour to call mom | timers that survive restarts and speak when due |
| turn off the living room light | Home Assistant, if you enable it and map names |

**How it is kept safe:**

- The grammar only ever sees *your* words. Model output and fetched web text are never parsed as
  commands, so a hallucination or a hostile page cannot lock your screen.
- Patterns are anchored: "what is the volume of a sphere" does nothing.
- "open X" launches only apps in the config's allow-list. Commands are argument lists, never shell
  strings. URLs must be http(s) with a real hostname, no credentials or ports.
- Lock, sleep, restart and shutdown always ask for a spoken or clicked yes (enforced inside the controller,
  not just the UI). It expires after `confirm_ttl_s`, any other message cancels it, and it is per session.
  Restart and shutdown are refused outright unless `allow_power_actions: true`. The yes is checked **again**
  against the caller's right to control the device: the confirmation protects against mishearing, it is not an
  access check, so someone who may not control the machine cannot complete an action somebody else started.
- Over HTTP, control works only for loopback callers or ones with the API key, and never for a request
  carrying another website's `Origin` (so a page open in your browser can't press buttons). Requests that
  pass through a reverse proxy are never treated as local; use an API key there.
- **DNS rebinding.** A web page can make your browser re-point its own domain at `127.0.0.1`; its requests then
  arrive with `Host: evil.example` *and* `Origin: http://evil.example`, which agree with each other. So "Origin
  equals Host" is not proof of "served by this server". The `Host` header is checked against an allow-list
  (`localhost`, any IP address, and names you list in `ZIVAWHITE_ALLOWED_HOSTS`) for **every** request, and a
  request with another name is refused with 400 before anything runs.
- **Conversation ids are issued by the server.** A pending "are you sure?" lives under a session id. HTTP callers
  no longer choose it (a client-chosen id could name the voice loop's `voice` or the CLI's `cli` session): the
  server returns a random id in each reply, ids it did not issue are ignored, and HTTP sessions are namespaced
  (`http:<id>`) so they can never collide with the local ones.
- Platform notes: Linux needs the usual tools (`pactl`/`wpctl`/`amixer`, `brightnessctl`, `playerctl`,
  a screenshot tool); a missing one is named in the reply. macOS brightness needs `brew install brightness`;
  lock uses a System Events keystroke, which needs Accessibility permission. On Windows volume is exact only
  with `pycaw` (otherwise approximated with volume keys), and brightness works on laptop panels.
