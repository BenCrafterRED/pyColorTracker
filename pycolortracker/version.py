from typing import Tuple


VERSION = (0, 0, 1)
STATE = "prealpha"


def get_version_tuple() -> Tuple[int, int, int]:
    return VERSION

def get_release_state() -> str:
    return STATE

def get_major_version() -> int:
    return VERSION[0]

def get_minor_version() -> int:
    return VERSION[1]

def get_patch_version() -> int:
    return VERSION[2]

def get_version_string(include_release_state: bool = False) -> str:
    return f"{VERSION[0]}.{VERSION[1]}.{VERSION[2]}{(STATE if include_release_state else '')}"
