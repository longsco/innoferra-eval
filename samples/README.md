# Synthetic bypass samples

Partner-runnable stand-ins for real traffic. Generated (seed 20260922) to match the **shape mix** of a 300-request survey of
production MiniMax-M3 traffic — root role 99%, tools 64%, streaming 67%, thinking adaptive 70% / disabled 29%, images 12%,
multi-turn agentic histories with `tool_calls` and `tool` messages, size buckets from 1k to 180k chars — with generated text only.
No customer content. Regenerate with the script in git history (`samples/` generator) if the mix changes.

Real captures (`innoferra capture`) are internal-only, land in `results/captures/`, and are gitignored.
