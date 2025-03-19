import random
from enum import Enum


class Command(Enum):
    HEAD_UP = 1
    HEAD_DOWN = 2
    HEAD_LEFT = 3
    HEAD_RIGHT = 4
    HEAD_TILTING_LEFT = 5
    HEAD_TILTING_RIGHT = 6
    EYE_BLINK = 7
    MOUTH_OPEN = 8
    PORTRAIT = 9
    SMILE = 10


LIST_COMMAND = [e for e in Command]


def check_command(video, command, **kwargs):
    """
    Process in short video with command is defined
    :param video: estimate 3s for length. This is impossible because user can wait
    :param command:
    :param kwargs:
    :return:
    """
    pass


def gen_command(n=3):
    """
    Generate random sequence command from list command
    :param n: number of commands want to generate
    :return:
    """
    return random.choices(LIST_COMMAND, k=n)


def get_command(cmd: str) -> Command:
    return Command[cmd]


def get_str_command(cmds):
    return [Command[cmd] for cmd in cmds]


def get_int_command(cmds):
    return [Command(cmd) for cmd in cmds]


def next_command():
    """
    Get random command
    :return:
    """
    return random.choice(LIST_COMMAND)
