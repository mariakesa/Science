#!/usr/bin/env python3
"""
Simple image labeling GUI for Allen natural scene images.

Default image folder:
    /home/maria/Science/data/images

Default output:
    /home/maria/Science/data/image_labels.npy

Label convention:
    -1 = unlabeled
     0 = inanimate
     1 = animate

The saved .npy file is a dictionary:
    data = np.load("image_labels.npy", allow_pickle=True).item()
    labels = data["labels"]
    image_paths = data["image_paths"]
    label_names = data["label_names"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

LABEL_NAMES = {
    -1: "unlabeled",
     0: "inanimate",
     1: "animate",
}

LABEL_NAMES={
    -1: "unlabeled",
     0: "everything-else",
     1: "scenes",
}


class ImageLabeler:
    def __init__(self, root: tk.Tk, image_dir: Path, output_path: Path, max_width: int = 1000, max_height: int = 750):
        self.root = root
        self.image_dir = image_dir.expanduser().resolve()
        self.output_path = output_path.expanduser().resolve()
        self.max_width = max_width
        self.max_height = max_height

        self.image_paths = self._find_images(self.image_dir)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        self.labels = np.full(len(self.image_paths), -1, dtype=np.int64)
        self.index = 0
        self.photo = None

        self._load_existing_labels()
        self._build_ui()
        self._bind_keys()
        self._show_current_image()

    def _find_images(self, image_dir: Path) -> list[Path]:
        return sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )

    def _load_existing_labels(self) -> None:
        if not self.output_path.exists():
            return

        try:
            saved = np.load(self.output_path, allow_pickle=True).item()
            saved_paths = [Path(p).name for p in saved["image_paths"]]
            current_names = [p.name for p in self.image_paths]

            if saved_paths == current_names:
                old_labels = np.asarray(saved["labels"], dtype=np.int64)
                if old_labels.shape == self.labels.shape:
                    self.labels = old_labels
                    first_unlabeled = np.where(self.labels == -1)[0]
                    self.index = int(first_unlabeled[0]) if len(first_unlabeled) else 0
                else:
                    messagebox.showwarning(
                        "Label file mismatch",
                        "Existing label file has a different shape. Starting a fresh label array.",
                    )
            else:
                messagebox.showwarning(
                    "Image list changed",
                    "Existing label file image names do not match current folder. Starting a fresh label array.",
                )
        except Exception as exc:
            messagebox.showwarning(
                "Could not load existing labels",
                f"Could not read {self.output_path}.\n\nStarting fresh.\n\nError: {exc}",
            )

    def _build_ui(self) -> None:
        self.root.title("Image Labeler: animate vs inanimate")

        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=8)

        self.status_label = tk.Label(self.top_frame, text="", font=("Arial", 13))
        self.status_label.pack(side=tk.LEFT)

        self.progress_label = tk.Label(self.top_frame, text="", font=("Arial", 11))
        self.progress_label.pack(side=tk.RIGHT)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Button(self.button_frame, text="← Previous", command=self.previous_image, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Inanimate [0]", command=lambda: self.set_label(0), width=16).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Animate [1]", command=lambda: self.set_label(1), width=16).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Clear label [C]", command=self.clear_label, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Next →", command=self.next_image, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(self.button_frame, text="Save [S]", command=self.save_labels, width=12).pack(side=tk.RIGHT, padx=4)

        self.help_label = tk.Label(
            self.root,
            text="Keys: 0=inanimate, 1=animate, ←/→=navigate, space=next, c=clear, s=save, q=quit",
            font=("Arial", 10),
        )
        self.help_label.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda event: self.previous_image())
        self.root.bind("<Right>", lambda event: self.next_image())
        self.root.bind("<space>", lambda event: self.next_image())
        self.root.bind("0", lambda event: self.set_label(0))
        self.root.bind("1", lambda event: self.set_label(1))
        self.root.bind("c", lambda event: self.clear_label())
        self.root.bind("C", lambda event: self.clear_label())
        self.root.bind("s", lambda event: self.save_labels())
        self.root.bind("S", lambda event: self.save_labels())
        self.root.bind("q", lambda event: self.quit())
        self.root.bind("Q", lambda event: self.quit())

    def _show_current_image(self) -> None:
        path = self.image_paths[self.index]

        image = Image.open(path).convert("RGB")
        image.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=self.photo)

        label_value = int(self.labels[self.index])
        label_text = LABEL_NAMES.get(label_value, f"unknown label {label_value}")

        self.status_label.configure(
            text=f"{self.index + 1}/{len(self.image_paths)}  {path.name}  |  label: {label_text}"
        )

        n_labeled = int(np.sum(self.labels != -1))
        n_inanimate = int(np.sum(self.labels == 0))
        n_animate = int(np.sum(self.labels == 1))
        self.progress_label.configure(
            text=f"Labeled: {n_labeled}/{len(self.labels)} | inanimate={n_inanimate}, animate={n_animate}"
        )

    def set_label(self, value: int) -> None:
        self.labels[self.index] = value
        self.save_labels(show_popup=False)

        if self.index < len(self.image_paths) - 1:
            self.index += 1

        self._show_current_image()

    def clear_label(self) -> None:
        self.labels[self.index] = -1
        self.save_labels(show_popup=False)
        self._show_current_image()

    def previous_image(self) -> None:
        if self.index > 0:
            self.index -= 1
        self._show_current_image()

    def next_image(self) -> None:
        if self.index < len(self.image_paths) - 1:
            self.index += 1
        self._show_current_image()

    def save_labels(self, show_popup: bool = True) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "image_dir": str(self.image_dir),
            "image_paths": np.array([str(p) for p in self.image_paths], dtype=object),
            "image_names": np.array([p.name for p in self.image_paths], dtype=object),
            "labels": self.labels.astype(np.int64),
            "label_names": LABEL_NAMES,
        }
        np.save(self.output_path, payload, allow_pickle=True)

        # Also save a small human-readable sidecar file because future-you deserves kindness.
        sidecar_path = self.output_path.with_suffix(".json")
        sidecar_payload = {
            "image_dir": str(self.image_dir),
            "output_path": str(self.output_path),
            "label_convention": {str(k): v for k, v in LABEL_NAMES.items()},
            "items": [
                {"image": p.name, "path": str(p), "label": int(label)}
                for p, label in zip(self.image_paths, self.labels)
            ],
        }
        sidecar_path.write_text(json.dumps(sidecar_payload, indent=2), encoding="utf-8")

        if show_popup:
            messagebox.showinfo("Saved", f"Saved labels to:\n{self.output_path}\n\nAlso wrote:\n{sidecar_path}")

    def quit(self) -> None:
        self.save_labels(show_popup=False)
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("/home/maria/Science/data/images"),
        help="Directory containing images such as scene_000.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        #default=Path("/home/maria/Science/data/image_labels.npy"),
        default=Path("/home/maria/Science/data/scenes_vs_Everything_labels.npy"),
        help="Where to save the .npy labels file",
    )
    parser.add_argument("--max-width", type=int, default=1000)
    parser.add_argument("--max-height", type=int, default=750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    app = ImageLabeler(
        root=root,
        image_dir=args.image_dir,
        output_path=args.output,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    root.protocol("WM_DELETE_WINDOW", app.quit)
    root.mainloop()


if __name__ == "__main__":
    main()
