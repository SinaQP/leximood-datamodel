# Contributing

Thanks for your interest in improving LexiMood.

## Development setup
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Suggested workflow
1. Create a feature branch.
2. Keep changes focused and small.
3. Run local checks before opening a PR.
4. Open a pull request with a clear summary and test notes.

## Local checks
```bash
python -m compileall src tests
pytest
```

## Style guidelines
- Prefer clear names over abbreviations.
- Add comments only when intent is not obvious from code.
- Keep notebook and script behavior aligned when modifying training logic.
