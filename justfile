# Default command when 'just' is run without arguments
# Run 'just <command>' to execute a command.
default: list

# Display help
help:
  @printf "\nSee Makefile targets for just and direnv installation."
  @printf "\nRun 'just -n <command>' to print what would be executed...\n\n"
  @just --list --unsorted
  @echo "\n...by running 'just <command>'.\n"
  @echo "This message is printed by 'just help'."
  @echo "Just 'just' will just list the available recipes.\n"

# List just recipes
list:
  @just --list --unsorted

# List evaluated just variables
vars:
  @just --evaluate

# Run tests
test:
  poetry run pytest -rA

# Run tests with coverage
test-cov-xml:
  poetry run pytest -rA --cov-report=xml

# Run linter
lint:
  poetry run black .
  poetry run ruff --fix .

# Run linter in check mode
lint-check:
  poetry run black --check .
  poetry run ruff .

# Build documentation
docs-build:
  poetry run mkdocs build

# Serve documentation
docs-serve: docs-build
  poetry run mkdocs serve

# e.g.: just remove-widgets "docs/notebooks/nsde_example.ipynb"
# https://github.com/jupyter/nbconvert/issues/1731#issuecomment-1157006081
# Remove widgets metadata from notebooks
remove-widgets notebook_path:
  jq -M 'del(.metadata.widgets)' {{notebook_path}} > {{notebook_path}}.tmp && mv {{notebook_path}}.tmp {{notebook_path}}

# Sync jupyter notebooks with text formats
nbsync:
  poetry run jupytext --sync **/*.ipynb

# Get the Co-authored-by string for a given GitHub username
coauthored-by github_username:
	@ID=$(curl -s https://api.github.com/users/{{github_username}} | jq '.id') && \
	echo "Co-authored-by: {{github_username}} <$ID+{{github_username}}@users.noreply.github.com>"
