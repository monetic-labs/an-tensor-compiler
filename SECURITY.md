# Security Policy

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.2.x   | Yes       |
| < 0.2   | No — please upgrade |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.** Use this
repo's confidential channel instead:

**Security tab → Report a vulnerability** (GitHub private vulnerability
reporting). Reports go directly to the maintainers.

Include in your report:

- What is affected (module, function, version, feature flags).
- What an attacker can achieve (panic, wrong results, memory unsafety…).
- A minimal reproducer (test case or snippet) if you have one.
- Whether the issue needs `metal`/`cuda` features or triggers on stock CPU.

## What to Expect

- Acknowledgment of receipt.
- Assessment and a fix timeline once confirmed.
- A patched release plus credit in the release notes (unless you prefer
  to stay anonymous).

## Scope Notes

This crate executes user-supplied rule text as tensor programs. Denial of
service via pathological inputs (extreme rule counts, dimensions) is a
known resource-limits area — report it if you find something
disproportionate (e.g. small input, unbounded memory). Unsafe code is not
used by this crate; any soundness issue in our safe-Rust surface is in
scope, as are dependency vulnerabilities reachable through our API.
