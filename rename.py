import os
import argparse

ignored_dirs = ["./.git", "./.idea"]
ignored_files = ["rename.py", ".DS_Store"]

replacements = {
    "year": "2021",
    "name": "psa",
    "name_punctuated": "PSA: Predictable Subspace Analysis",
    "author_first_name": "Wessel",
    "author_last_name": "Bruinsma",
    "author_email": "wessel.p.bruinsma@gmail.com",
    "repo": "wesselb/psa",
    "docs_domain": "wesselb.github.io/psa",
    "description": "Predictable subspace analysis",
}

# Prefix keys in `replacements` with `skeleton_`, and convert to list.
replacements = [("skeleton_" + k, v) for k, v in replacements.items()]

# Order based on length to deal with simple prefixes.
replacements = sorted(replacements, key=lambda x: len(x[0]), reverse=True)


def perform_replacements(x):
    """Perform replacements.

    Args:
        x (str): Input to perform replacements on.

    Returns:
        tuple: Tuple containing `x` with all replacements applied and a
            boolean indicating whether any replacement has happened.
    """
    any_replaced = False
    for k, v in replacements:
        if k in x:
            any_replaced = True
            x = x.replace(k, v)
    return x, any_replaced


def dir_begins_with(x, y):
    """Check if the path `x` begins with directory `y`.

    Args:
        x (str): Path to check.
        y (str): Beginning directory of `x`.

    Returns:
        bool: `x` begins with directory `y`.
    """
    x_parts = x.lower().split(os.sep)
    y_parts = y.lower().split(os.sep)
    return len(x_parts) >= len(y_parts) and x_parts[: len(y_parts)] == y_parts


def list_files(base_dir="."):
    """List all files and directories in a directory.

    Ignores ignored directories and ignored files.

    Args:
        base_dir (str, optional): Base directory. Defaults to `.`.

    Returns:
        list: List of all files and directories.
    """
    fs = []
    for path, dirs, files in os.walk(base_dir):
        # Don't walk through ignore directories.
        if any(dir_begins_with(path, d) for d in ignored_dirs):
            continue

        # Don't list ignores files or directories.
        fs.extend(
            [
                os.path.join(path, f)
                for f in files + dirs
                if not (os.path.isfile(f) and f in ignored_files)
                if not (
                    os.path.isdir(f)
                    and any(dir_begins_with(path, d) for d in ignored_dirs)
                )
            ]
        )

    # Sort with longest first to allow renaming.
    return sorted(fs, key=lambda x: len(x), reverse=True)


parser = argparse.ArgumentParser()
parser.add_argument("-w", "--wet", action="store_true", help="overwrite files")
args = parser.parse_args()

for file in list_files():
    if os.path.isfile(file):
        # Print the current file, and read the file.
        print("  File: {}".format(file))
        with open(file, "r") as f:
            content = f.read()

        # Perform replacements and show an overview.
        ln = 0
        lines = []
        for line in content.splitlines():
            ln += 1
            line, any_replaced = perform_replacements(line)
            if any_replaced:
                print("{:6d}: {}".format(ln, line))
            lines.append(line)

        # If wet run, then overwrite.
        if args.wet:
            with open(file, "w") as f:
                f.write("\n".join(lines))

    # Check for renaming in the basename.
    dirname = os.path.dirname(file)
    basename, any_replaced = perform_replacements(os.path.basename(file))
    if any_replaced:
        new_file = os.path.join(dirname, basename)
        print("Rename:", file, "->", new_file)

        # If wet, perform renaming.
        if args.wet:
            os.rename(file, new_file)
