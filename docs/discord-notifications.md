# Discord notifications

The `Notify FailureLens Discord` GitHub Actions workflow sends selected repository
events to the private FailureLens coordinator channel. It requires the repository
Actions secret `FAILURELENS_DISCORD_WEBHOOK_URL`, containing a Discord webhook URL
created for that channel. Never put the URL in the repository, an issue, or a PR.

The workflow sends one message for each newly opened PR and for each new PR
conversation comment, line comment, or non-empty review summary authored by the
observed Copilot GitHub identities. It does not send ordinary user comments, issue
comments outside a PR, PR updates, or empty review approvals. It sends metadata
and the GitHub URL, not the comment body; the coordinator must fetch the original
on GitHub before acting.

The coordinator's Discord routing and thread-creation instructions are maintained in OpenClaw,
outside this repository. The webhook targets the existing text channel; after a
notification arrives, the coordinator creates a separate thread from that message
for its review and follow-up. The channel session still receives the initial
notification and may need periodic compaction.

To verify the integration, check the workflow run for an eligible new PR or Copilot
comment, then confirm one webhook message in the coordinator channel and a new
coordinator thread. Also check that a non-Copilot PR comment produces no message.
