---
name: code-formatter
description: Formats Python code using black
version: 1.0.0
metadata:
  openclaw:
    requires:
      bins:
        - black
---

Format the user's Python files using the `black` code formatter.
Only format files explicitly requested by the user.
