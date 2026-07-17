"""Internal helpers for publishing complete, no-clobber artifacts."""

from __future__ import annotations

import os
import secrets
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO


def prospective_path_key(path: Path) -> tuple[str, ...]:
    """Return a conservative key for unresolved case/Unicode aliases."""

    return tuple(
        unicodedata.normalize(
            "NFC",
            unicodedata.normalize("NFC", component).casefold(),
        )
        for component in path.parts
    )


def is_prospective_ancestor(
    ancestor: tuple[str, ...],
    descendant: tuple[str, ...],
) -> bool:
    """Return whether one normalized prospective path contains another."""

    return (
        len(ancestor) < len(descendant)
        and descendant[: len(ancestor)] == ancestor
    )


def _create_temporary_file(path: Path) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    while True:
        temporary_path = path.with_name(
            f".{path.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(temporary_path, flags, 0o666)
        except FileExistsError:
            continue
        return descriptor, temporary_path


def write_new_file_atomically(
    path: Path,
    *,
    writer: Callable[[BinaryIO], None],
    exists_message: str,
) -> Path:
    """Encode to a same-directory temporary file and atomically link it."""

    descriptor, temporary_path = _create_temporary_file(path)
    try:
        handle = os.fdopen(descriptor, "wb")
        descriptor = -1
        with handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise FileExistsError(exists_message) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
    return path
