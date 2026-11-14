# Contributing

Issues are welcome: bug reports, questions, ideas, and requests to remove an image from the dataset. Pull requests are welcome too; for anything larger than a small fix, open an issue first.

Set up and check your change:

```sh
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests
python -m ruff check .
```

The tests use temporary data and mocked clients: they need no network, no GPU and no API key. Please keep them passing, add a test for new behaviour, and never commit data, weights, videos or keys (see `.gitignore`).
