"""

A module for matching structureed unlabelled cross sucking clip names to unstructured labelled cross sucking clip names (CVAT outputs).

"""

import re
from typing import List, Tuple, Optional, Union


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
    # Extract Numeric ID
    # Search for different part formats
    # Extract correct part location
    # Return Tuple
    pass


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
    FormatError
        If string is not of one of the known types.

    Notes
    -----

     Examples
    --------
    >>> parse_unlabelled_name("0001.zip")
    >>> ("1", None)

    >>> parse_unlabelled_name("0002_part01.mp4")
    >>> ("2", 1)

    >>> parse_unlabelled_name("0003 - p2.zip")
    >>> ("3", 2)

    >>> parse_unlabelled_name("0004_part02zip.zip")
    >>> ("4", 2)
    """
    # Extract Numeric ID
    # Search for different part formats
    # Extract correct part location
    # Return Tuple
    pass


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

    Raises
    ------
    ValueError
        If name is an empty string
    TypeError
        If name is not a string
    ValueError
        If no Numeric ID was found, or ID was not in a known format.


    Exampes
    >>> extract_numeric_id("0001.zip") >>> "1"
    >>> extract_numeric_id("0002_part01.zip") >>> "2"
    >>> extract_numeric_id("CS_0501_WEAN_d2_p2_cow6_17102025_ch02-20251017082112_3507_3577.mp4.zip") >>> "501"
    >>> extract_numeric_id("011.zip") >>> "11"
    >>> extract_numeric_id("01133333.zip") >>> 1133333
    """
    # Check type, non-empty
    # Check against different formats
    # Pull correct location from correct format
    # Return Numeric ID.

    pass


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
        The name of a single labelled cross sucking clip.
    None
        If no labelled clip corresponding to the input is found

    Raises
    ------
    FileNotFoundError
        If the path to the unlabelled video clip does not exist.?
    ValueError
        If there is an unaccounted for version in the labelled clip names?

    Notes
    -----

    Examples
    --------
    unlabelled_name = CS_0001_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702.mp4
    unlabelled_name_2 = CS_0009_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part02.mp4
    unlabelled_name_3 = CS_0101_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4
    unlabelled_name_34 = CS_9999_POSTWEAN_d1_p2_cow6_02112025_ch02-20251102075200_684_702_part01.mp4

    labelled_names = ["0001.zip", "0002.zip", ..., "0009_part01.zip", "0009_part02.zip", "0101_part01zip.zip"]

    >>> find_match(unlabelled_name, labelled_names) >>> "0001.zip"
    >>> find_match(unlabelled_name_2, labelled_names) >>> "0009_part02.zip"
    >>> find_match(unlabelled_name_3, labelled_names) >>> "0101_part01zip.zip"
    >>> find_match(unlabelled_name_4, labelled_names) >>> None
    """

    ### extract numeric id and part number from unlabelled and labelled names
    ### Check for matches
    ### Return match if exists, else return None

    return ""
