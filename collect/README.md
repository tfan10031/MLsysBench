
**⛏️ GitHub Data Collection**
* `print_pulls.py`
    * Purpose: Given the `<owner/name>` of a GitHub repo, this script writes the raw information for all the repo's PRs to a single `.jsonl` file
    * Usage: `python print_pulls.py <repo name> <path to PRs .jsonl file> --token <GitHub Token>`
* `build_dataset.py`
    * Purpose: Given the path to a PRs `.jsonl` file generated by `print_pulls.py`, this script attempts to convert each PR to a task instance. It creates a `jsonl.all` file for any PRs with an issue and a `.jsonl` file for any PRs with both an issue and modifications to that repository's tests.
    * Usage: `python build_dataset.py <path to PRs .jsonl file> <path to output .jsonl file> --token <Github Token>`

