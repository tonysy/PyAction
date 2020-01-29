#!/usr/bin/python3
# -*- coding:utf-8 -*-
import re
from colorama import Back, Fore, Style


def highlight(keyword: str, target: str, color=Fore.BLACK + Back.YELLOW) -> str:
    return re.sub(keyword, color + r"\g<0>" + Style.RESET_ALL, target)


def find_key(param_dict: dict, key: str) -> dict:
    find_result = {}
    for k, v in param_dict.items():
        if re.search(key, k):
            find_result[k] = v
        if isinstance(v, dict):
            res = find_key(v, key)
            if res:
                find_result[k] = res
    return find_result


def diff_dict(dict1: dict, dict2: dict) -> dict:
    diff_result = {}
    for k, v in dict1.items():
        if k not in dict2:
            diff_result[k] = v
        elif dict2[k] != v:
            if isinstance(v, dict):
                diff_result[k] = diff_dict(v, dict2[k])
            else:
                diff_result[k] = v
    return diff_result
