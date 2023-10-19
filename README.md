# tctsti

<p align="center">
    <em>Library for benchmarking the impact of transcript count type stratification on trajectory inference.</em>
</p>

[![build](https://github.com/pyrovelocity/tctsti/workflows/CI/badge.svg)](https://github.com/pyrovelocity/tctsti/actions)
[![codecov](https://codecov.io/gh/pyrovelocity/tctsti/branch/main/graph/badge.svg?kill_cache=1)](https://codecov.io/gh/pyrovelocity/tctsti)
[![PyPI version](https://badge.fury.io/py/tctsti.svg?branch=main&kill_cache=1)](https://badge.fury.io/py/tctsti)

---

**Documentation**: <a href="https://pyrovelocity.github.io/tctsti" target="_blank">https://pyrovelocity.github.io/tctsti</a>

**Source Code**: <a href="https://github.com/pyrovelocity/tctsti" target="_blank">https://github.com/pyrovelocity/tctsti</a>

---

## Development

### Setup environment

Copy the contents of [.example.env](./.example.env) to `.env`, review its contents, and update the variable values for your environment.

We use [direnv](https://direnv.net/), [just](https://just.systems/), and [poetry](https://python-poetry.org/docs/#installation) to manage the development environment. Please see the [Makefile](./Makefile) for installation instructions. You may be able to setup your environment removing `-n` from

```bash
make -n install_direnv install_just install_poetry
```

but you should review the output of that command and the contents of the [Makefile](./Makefile) before executing this in your environment.

After this is complete you may need to run `direnv --help` and `direnv allow` prior to `just help` or just `just`.

### Run unit tests

You can run the tests with:

```bash
just test
```

or produce the coverage report with `just test-cov-xml`.

### Format the code

Execute the following command to apply linting and check typing:

```bash
just lint
```

or check if changes are required first with `just link-check`.

## Serve the documentation

You can serve the mkdocs documentation with:

```bash
just docs-serve
```

This will automatically watch for code changes.

### Publish a new version

You can check the current version with:

```bash
poetry version
```

You can bump the version with commands that contain a bump rule

> The new version should ideally be a valid semver string or a valid bump rule:
> patch, minor, major, prepatch, preminor, premajor, prerelease.

such as `poetry version prepatch`. You can also simply edit the version in the[pyproject.toml](./pyproject.toml) file. After changing the version, when you push to github, the [CD](./.github/workflows/cd.yml) workflow will automatically publish it on Test-PyPI and a github release will be created as a draft.

## License

This project is licensed under the terms of the GNU Affero 3.0-only license.
