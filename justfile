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

# test = "pytest -rA"
# test-cov-xml = "pytest -rA --cov-report=xml"
# lint = "bash -c 'black . && ruff --fix .'"
# lint-check = "bash -c 'black --check . && ruff .'"
# docs-serve = "mkdocs serve"
# docs-build = "mkdocs build"

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
