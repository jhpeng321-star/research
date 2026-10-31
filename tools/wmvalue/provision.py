"""Create only the task-owned output directory under the verified task root."""
from pathlib import Path


def main():
    root = Path('/root/wmvalue20261031')
    assert root.is_dir()
    (root/'cells_v1').mkdir(exist_ok=True)
    print('output parent ready', root/'cells_v1', flush=True)


if __name__ == '__main__':
    main()
