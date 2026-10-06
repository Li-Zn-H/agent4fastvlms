# Security

This local research framework launches JSON-selected Python Workers and loads
model/backend code. Use trusted configurations, dependencies and checkpoints;
an isolated environment is not a security sandbox.

Report vulnerabilities through the hosting project's private channel when
available. Otherwise request a private contact without posting exploit details,
credentials or private data in a public issue. Include the affected revision,
versions, impact and a small sanitized reproduction. A dedicated reporting
channel remains a [release prerequisite](docs/releasing.md).

Keep tokens outside the repository. Review generated answers, image inputs and
receipts before sharing. GPU locks coordinate execution; host permissions control
access. Do not bypass backend preflight by enabling unsafe loading or remote code.
