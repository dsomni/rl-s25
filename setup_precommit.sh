#!/bin/bash
echo "#!/bin/sh
uv sync
uv run ruff format .
uv run ruff check . --fix --exit-non-zero-on-fix
FILES=\$(git diff --diff-filter=d --name-only --staged)
git add \$FILES
" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit