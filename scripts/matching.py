"""

A module for matching structured unlabelled cross sucking clip names to
unstructured labelled cross sucking clip names (CVAT outputs).

"""

import re
from typing import List, Tuple, Optional


def parse_unlabelled_name(unlabelled_name: str) -> Tuple[str, Optional[int]]:
    """
    Parses an unlabelled clip name into Numeric ID and Part number

    Parameters
    ----------
    unlabelled_name : str
        The name of a single unlabelled cross sucking clip.

    Returns
    -------
    Tuple[str, Optional[int]]
        A tuple containing the Numeric ID (str) and part number (int) if it exists.

    Raises
    ------
    InputError
        If unlabelled_name is an empty string
    TypeError
        If unlabelled_name is not a string

    Notes
    -----

    Examples
    --------
    >>> parse_unlabelled_name("CS_0005_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4")
    >>> ("5", None)

    >>> parse_unlabelled_name("CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part01.mp4")
    >>> ("6", 1)

    >>> parse_unlabelled_name("CS_0006_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577_part02.mp4")
    >>> ("6", 2)
    """
    if not isinstance(unlabelled_name, str):
        raise TypeError(f"Expected string, got {type(unlabelled_name).__name__}")
    if not unlabelled_name.strip():
        raise ValueError("Filename cannot be empty or whitespace only")

    # Remove video file extensions
    name = re.sub(r"\.(mp4|avi|mov|mkv|MP4|AVI|MOV|MKV)$", "", unlabelled_name)

    # Extract part number if present
    part_match = re.search(r"_part(\d+)$", name, re.IGNORECASE)

    if part_match:
        part_num = int(part_match.group(1))
        numeric_id = extract_numeric_id(name)
    else:
        part_num = None
        numeric_id = extract_numeric_id(name)

    return numeric_id, part_num


def parse_labelled_name(labelled_name: str) -> Tuple[str, Optional[int]]:
    """
    Parses a labelled clip name into Numeric ID and part number.

    Parameters
    ----------
    labelled_name : str
        The name of a single labelled cross sucking clip.

    Returns
    -------
    Tuple[str, Optional[int]]
        A tuple containing the Numeric ID (str) and part number (int) if it exists.

    Raises
    ------
    InputError
        If labelled_name is an empty string.
    TypeError
        If labelled_name is not a string.
    ValueError
        If labelled name is not  of the known types.

    Notes
    -----

     Examples
    --------
    >>> parse_labelled_name("0001.zip")
    >>> ("1", None)

    >>> parse_labelled_name("0002_part01.mp4")
    >>> ("2", 1)

    >>> parse_labelled_name("0003 - p2.zip")
    >>> ("3", 2)

    >>> parse_labelled_name("0004_part02zip.zip")
    >>> ("4", 2)
    """
    # Remove archive extensions
    name = re.sub(r"\.(zip|rar|7z|ZIP|RAR|7Z)$", "", labelled_name)

    # Remove any remaining periods: 1234..zip
    name = re.sub(r"\.+$", "", name)

    # Also remove .mp4 if present before .zip
    name = re.sub(r"\.(mp4|avi|mov|mkv|MP4|AVI|MOV|MKV)$", "", name)

    # Try various part number patterns (in order of specificity)
    part_patterns = [
        r"(^\d+$)",  # Basic pattern with no parts: 0001
        r"_part(\d+)zip$",  # Typo: _part02zip
        r"_part(\d+)$",  # _part01, _part02, _part1, _part2
        r"_p(\d+)$",  # _p01, _p02, _p1, _p2
        r"-p(\d+)$",  # -p1, -p2
        r"\s+-\s+p(\d+)$",  # - p1, - p2 (with spaces)
        r"(?<!_)p(\d+)",  # 1234p1 (not hyphen or underscore)
        r"^CS_(\d+)",  # CC_XXXX pattern with no parts
    ]

    part_num = None

    for pattern in part_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            # Parse Part
            if (pattern == r"(^\d+$)") or (
                pattern == r"^CS_(\d+)"
            ):  # Basic format: 0001
                part_num = None
                base_name = name
            else:
                part_num = int(match.group(1))
                base_name = name[: match.start()]

            # Extract Numeric Id
            try:
                numeric_id = extract_numeric_id(base_name)

                # Retrun Numeric ID and Tuple
                return numeric_id, part_num
            # Try next pattern
            except ValueError:
                continue

    # Raise Error no patterns match
    raise ValueError(
        f"{labelled_name} does not match known naming patterns. See documentation for more infor on known naming conventions."
    )


def extract_numeric_id(name: str) -> str:
    """
    Extract a numeric if from given cross sucking clip name (labelled or
    unlabelled) and returns it as a string with leading zeroes removed.

    Parameters
    ----------
    name : str
        The name of an unlabelled or labelled cross sucking clip

    Returns
    -------
    str
        The four digit numeric ID representing the video number
    None
        If no numeric ID is found

    Raises
    ------
    ValueError
        If name is an empty string or if a numeric ID could not be found.
    TypeError
        If name is not a string


    Exampes
    >>> extract_numeric_id("0001.zip") >>> "1"
    >>> extract_numeric_id("0002_part01.zip") >>> "2"
    >>> extract_numeric_id("CS_0501_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip") >>> "501"
    >>> extract_numeric_id("011.zip") >>> "11"
    >>> extract_numeric_id("01133333.zip") >>> 1133333
    """

    # Check string type
    if not isinstance(name, str):
        raise TypeError(f"Expected a string, but received {type(name).__name__}")

    # Check non-empty
    if not name.strip():
        raise ValueError("Filename cannot be an empty string or whitespace only.")

    # Try CS_XXXX pattern
    cs_match = re.match(r"CS_(\d+)", name, re.IGNORECASE)
    if cs_match:
        return str(int(cs_match.group(1)))  # Remove leading zeros

    # Try leading number
    num_match = re.match(r"^(\d+)", name)
    if num_match:
        return str(int(num_match.group(1)))  # Remove leading zeros

    raise ValueError(f"Could not extract numeric id from {name}")


def find_match(unlabelled_name: str, labelled_names: List[str]) -> str | None:
    """
    Takes in a name from an unlabelled clip and finds the name of the corresponding labelled clip.

    Supports:
    - Full structured name matching (CS_XXX_... format)
        - Numeric ID extractuion and matching.
        - Flexible part matching between structured unlabelled clip names and unstructured
        labelled clip names (_part01, _part1, _p01, _p1, -p1, etc.).
        - Typo tolerance and format variations


    Parameters
    ----------
    unlabelled_name : str
        The name of a single unlabelled cross sucking clip.
    labelled_names : list[str]
        A list of all labelled cross sucking clips.

    Returns
    -------
    str
        The name of a single labelled cross sucking clip matching the unlabelled_name input.
    None
        If no labelled cross sucking clip corresponding to the unlabelled_name input is found

    Raises
    ------
    ValueError
        If inputs are empty strings or empty lists.
    TypeError
        If the inputs are not a string and a list of strings.

    Notes
    -----
    Right now, this is configured for .mp4, .avi, .mov, .mkv video extensions and .zip labelled data extensions.

    Examples
    --------
    unlabelled_name = CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4
    unlabelled_name_2 = CS_0009_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part02.mp4
    unlabelled_name_3 = CS_0101_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4
    unlabelled_name_4 = CS_9999_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4

    labelled_names = ["0001.zip", "0002.zip", ..., "0009_part01.zip", "0009_part02.zip", "0101_part01zip.zip"]

    >>> find_match(unlabelled_name, labelled_names) >>> "0001.zip"
    >>> find_match(unlabelled_name_2, labelled_names) >>> "0009_part02.zip"
    >>> find_match(unlabelled_name_3, labelled_names) >>> "0101_part01zip.zip"
    >>> find_match(unlabelled_name_4, labelled_names) >>> None
    """
    # Check string type
    if not isinstance(unlabelled_name, str):
        raise TypeError(
            f"Expected a string, but received {type(unlabelled_name).__name__}"
        )

    # Check list type
    if not isinstance(labelled_names, list):
        raise TypeError(
            f"Expected a list, but received {type(labelled_names).__name__}"
        )

    # Check list is not empty
    if not labelled_names:
        raise ValueError("labelled_names cannot be an empty list")

    # Check list of strings
    if not all(isinstance(name, str) for name in labelled_names):
        raise TypeError("Expected a list of strings")

    # Check non-empty str
    if not unlabelled_name.strip():
        raise ValueError("Filename cannot be an empty string or whitespace only.")

    # Parse unlabeled name
    try:
        parsed_ul = parse_unlabelled_name(unlabelled_name)
    except ValueError as e:
        raise ValueError(f"Failed to parse unlabelled name '{unlabelled_name}': {e}")
    except Exception as e:
        raise e  # Re-raise unexpected exceptions

    matches = []
    for name in labelled_names:

        # Parse Labelled name
        try:
            parsed_l = parse_labelled_name(name)
            if parsed_ul == parsed_l:
                matches.append(name)
        except Exception as e:
            raise e

    # There should be exactly one or zero matches
    if len(matches) > 1:
        raise ValueError(
            f"{unlabelled_name} matched to multiple labelled names: {matches}"
        )
    elif len(matches) == 1:
        return matches[0]
    else:
        return None


def is_match(unlabelled_name: str, labelled_name: str) -> bool:
    """
    Returns True if the unlabelled and labelled names match on numeric ID and part number

    Supports:
    - Full structured name matching (CS_XXX_... format)
        - Numeric ID extractuion and matching.
        - Flexible part matching between structured unlabelled clip names and unstructured
        labelled clip names (_part01, _part1, _p01, _p1, -p1, etc.).
        - Typo tolerance and format variations


    Parameters
    ----------
    unlabelled_name : str
        The name of a single unlabelled cross sucking clip.
    labelled_names : str
        The name of a single labelled cross sucking clip.

    Returns
    -------
    bool
        Return true if names match, false otherwise.

    Raises
    ------
    ValueError
        If inputs are empty.
    TypeError
        If the inputs are not strings.

    Notes
    -----
    Right now, this is configured for .mp4, .avi, .mov, .mkv video extensions and .zip labelled data extensions.

    Examples
    --------
    unlabelled_name = CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4
    unlabelled_name_2 = CS_0009_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part02.mp4
    unlabelled_name_3 = CS_0101_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4
    unlabelled_name_4 = CS_9999_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4

    >>> is_match(unlabelled_name, "0001.zip") >>> True
    >>> is_match(unlabelled_name_2, "0009_part01.zip") >>> True
    >>> is_match(unlabelled_name_3, "0101_part01zip.zip") >>> True
    >>> is_match(unlabelled_name_4, "0001.zip") >>> False
    """
    # Check string type
    if not isinstance(unlabelled_name, str):
        raise TypeError(
            f"Expected a string, but received {type(unlabelled_name).__name__}"
        )

    # Check list type
    if not isinstance(labelled_name, str):
        raise TypeError(
            f"Expected a string, but received {type(labelled_name).__name__}"
        )

    # Check non-empty str
    if not (unlabelled_name.strip() or labelled_name.strip()):
        raise ValueError("Filenames cannot be an empty string or whitespace only.")

    # Try Parse unlabeled name
    try:
        parsed_ul = parse_unlabelled_name(unlabelled_name)
    except ValueError as e:
        raise ValueError(f"Failed to parse unlabelled name '{unlabelled_name}': {e}")
    except Exception as e:
        raise e  # Re-raise unexpected exceptions

    # Try Parse Labelled name
    try:
        parsed_l = parse_labelled_name(labelled_name)
    except ValueError as e:
        raise ValueError(f"Failed to parse unlabelled name '{labelled_name}': {e}")
    except Exception as e:
        raise e  # Re-raise unexpected exceptions

    # Returns True if the numeric id and part match for both names
    return parsed_ul == parsed_l
