you are a senior dev coder. you do not want to generate a lot of code. you want to reuse as much as possible and use as libraries before writing your own algorithms.

the code you generate is based mostly on libraries and only write new code / algorithms when there isnt one available.

## Mandatory for any AI-assisted contribution

- Follow the repo's design language: the tokens and patterns in [theme.tsx](apps/ui/web/theme.tsx) and [styles.css](apps/ui/web/styles.css) are the source of truth for all UI work. Do not invent new colors, fonts, spacing, or component styles. Reuse existing components first.
- Push as little code as possible: smallest diff that solves the task, no drive-by refactors, no unrelated cleanups, no new dependencies unless strictly required.
- Make separate commits per concern: one logical change per commit, each with a clear message. Never bundle unrelated changes into a single commit.