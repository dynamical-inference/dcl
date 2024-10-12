import git

file_extensions_to_ignore = [
    ".ipynb",
    ".sh",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".csv",
    ".tsv",
    ".xlsx",
]


def has_untracked_changes(repo):
    untracked_files = repo.untracked_files
    # filter for files in dcl folder
    untracked_files = [file for file in untracked_files if "dcl/" in file]
    # filter out the .ipynb_checkpoints
    for ext in file_extensions_to_ignore:
        untracked_files = [file for file in untracked_files if ext not in file]
    return len(untracked_files) > 0


def has_uncommitted_changes(repo):
    diffs = repo.index.diff(None)
    # filter for files in dcl folder
    diffs = [diff for diff in diffs if "dcl/" in diff.a_path]
    for ext in file_extensions_to_ignore:
        diffs = [diff for diff in diffs if ext not in diff.a_path]
    return len(diffs) > 0


def check_repo_status():
    repo = git.Repo(search_parent_directories=True)
    if not has_untracked_changes(repo) and not has_uncommitted_changes(repo):
        return repo.head.object.hexsha
    else:
        raise ValueError("There are untracked or uncommitted files.")


if __name__ == "__main__":
    print(check_repo_status())
