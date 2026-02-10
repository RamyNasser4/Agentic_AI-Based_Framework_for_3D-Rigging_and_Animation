from pathlib import Path

# path to your folder
folder = Path(r"C:\Games\My-GitHub\GP\CMU\01")

def print_tree(path, prefix=""):
    items = sorted(path.iterdir())
    for i, item in enumerate(items):
        connector = "└── " if i == len(items)-1 else "├── "
        print(prefix + connector + item.name)
        if item.is_dir():
            extension = "    " if i == len(items)-1 else "│   "
            print_tree(item, prefix + extension)

print(folder.name)
print_tree(folder)
