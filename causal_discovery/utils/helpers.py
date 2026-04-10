"""
Shared utility functions for parsing LLM responses.
"""

import ast
import re


def parse_answer_tag(text):
    """
    Extract the content between <Answer>...</Answer> tags from LLM output.

    Args:
        text: raw LLM response string.

    Returns:
        str or None: the extracted answer, or None if no tag found.
    """
    if text is None:
        return None
    match = re.search(r'<Answer>(.*?)</Answer>', text)
    if match:
        return match.group(1).strip()
    return None


def str_2_lst(str1):
    """
    Convert a string representation of a list of tuples into an actual list.
    Single elements are wrapped in a 1-tuple.

    Example:
        "[('A','B'), ('C')]" -> [('A','B'), ('C',)]
    """
    lst1 = ast.literal_eval(str1)
    lst1 = [ele if isinstance(ele, tuple) else (ele,) for ele in lst1]
    return lst1


def str_filter_lst(str1):
    """
    Extract the first list or tuple from a string using regex.

    Example:
        "some text [('A','B')] more text" -> [('A','B')]
    """
    pattern = r"\[.*?\]|\(.*?\)"
    result = re.findall(pattern, str1)
    result = [eval(x) for x in result]
    return result[0]
