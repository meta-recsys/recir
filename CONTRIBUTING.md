# Contributing to meta-recsys/recir
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
We may periodically sync internal changes to this project, although we do not
currently expect this to be a frequent process except for major changes.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## CPU checks

The `CPU CI` GitHub Actions workflow runs on pushes and pull requests, and can
also be started manually. On Python 3.12 it checks Python syntax throughout
the source and tests, builds a source distribution and wheel, and validates
their package metadata. These checks run on a standard Ubuntu runner without
an accelerator or the model-training dependencies.

Run the same checks from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
export PYTHONUTF8=1
python -m pip install --upgrade pip
python -m pip install build twine
python -m pip check
python -m compileall -q recir setup.py
python -m build
python -m twine check --strict dist/*
```

This workflow does not import RecIR or run runtime unit tests. At the initial
CI baseline, `import recir` fails because of a circular import involving the
compiler, distributed, and XLA lowering packages. Runtime tests can be added
after that issue is fixed separately. The checks above validate syntax and
packaging; they do not establish training or GPU/TPU correctness. See the
[DLRMv3 guide](recir/examples/dlrmv3/README.md) for reference training commands.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* 2 spaces for indentation rather than tabs
* 80 character line length
* ...

## License
By contributing to meta-recsys/recir, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
