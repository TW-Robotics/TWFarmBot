# twfarmbot-ui

Dashboard, sensor display, and manual triggers.

This app is a thin orchestration layer on top of `twfarmbot-core`. It should
not contain domain logic; it only renders state and forwards user actions
to the API server.
