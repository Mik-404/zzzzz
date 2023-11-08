import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.lang import Builder
from kivy.core.text import LabelBase
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.clipboard import Clipboard

LabelBase.register(name='Jost', fn_regular='Jost-Light.ttf')
LabelBase.register(name='Oxygen', fn_regular='Oxygen-Bold.ttf')

import collections
import collections.abc
import functools
import itertools
import math
import operator
import types as _types
import inspect
import os
import re
import struct
import sys
import warnings
import zlib
import abc

from array import array
from bisect import bisect_left
from typing import Dict, Generic, List, NamedTuple, Optional, Type, TypeVar, cast, overload
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Type, Union
import typing
from PIL import Image, ImageDraw
from itertools import chain
import types as _types

import numpy as np
import cv2 as cv
from pyzbar import pyzbar
from kivy.logger import Logger
from kivy.clock import Clock

from tkinter.filedialog import askopenfilename
from tkinter import Tk
import tkinter.filedialog as fd


if 1:
    Literal = typing.Literal

    EXP_TABLE = list(range(256))

    LOG_TABLE = list(range(256))

    for i in range(8):
        EXP_TABLE[i] = 1 << i

    for i in range(8, 256):
        EXP_TABLE[i] = (
            EXP_TABLE[i - 4] ^ EXP_TABLE[i - 5] ^ EXP_TABLE[i - 6] ^ EXP_TABLE[i - 8]
        )

    for i in range(255):
        LOG_TABLE[EXP_TABLE[i]] = i

    RS_BLOCK_OFFSET = {1: 0, 0: 1, 3: 2, 2: 3}

    RS_BLOCK_TABLE = (
        # L
        # M
        # Q
        # H
        # 1
        (1, 26, 19),
        (1, 26, 16),
        (1, 26, 13),
        (1, 26, 9),
        # 2
        (1, 44, 34),
        (1, 44, 28),
        (1, 44, 22),
        (1, 44, 16),
        # 3
        (1, 70, 55),
        (1, 70, 44),
        (2, 35, 17),
        (2, 35, 13),
        # 4
        (1, 100, 80),
        (2, 50, 32),
        (2, 50, 24),
        (4, 25, 9),
        # 5
        (1, 134, 108),
        (2, 67, 43),
        (2, 33, 15, 2, 34, 16),
        (2, 33, 11, 2, 34, 12),
        # 6
        (2, 86, 68),
        (4, 43, 27),
        (4, 43, 19),
        (4, 43, 15),
        # 7
        (2, 98, 78),
        (4, 49, 31),
        (2, 32, 14, 4, 33, 15),
        (4, 39, 13, 1, 40, 14),
        # 8
        (2, 121, 97),
        (2, 60, 38, 2, 61, 39),
        (4, 40, 18, 2, 41, 19),
        (4, 40, 14, 2, 41, 15),
        # 9
        (2, 146, 116),
        (3, 58, 36, 2, 59, 37),
        (4, 36, 16, 4, 37, 17),
        (4, 36, 12, 4, 37, 13),
        # 10
        (2, 86, 68, 2, 87, 69),
        (4, 69, 43, 1, 70, 44),
        (6, 43, 19, 2, 44, 20),
        (6, 43, 15, 2, 44, 16),
        # 11
        (4, 101, 81),
        (1, 80, 50, 4, 81, 51),
        (4, 50, 22, 4, 51, 23),
        (3, 36, 12, 8, 37, 13),
        # 12
        (2, 116, 92, 2, 117, 93),
        (6, 58, 36, 2, 59, 37),
        (4, 46, 20, 6, 47, 21),
        (7, 42, 14, 4, 43, 15),
        # 13
        (4, 133, 107),
        (8, 59, 37, 1, 60, 38),
        (8, 44, 20, 4, 45, 21),
        (12, 33, 11, 4, 34, 12),
        # 14
        (3, 145, 115, 1, 146, 116),
        (4, 64, 40, 5, 65, 41),
        (11, 36, 16, 5, 37, 17),
        (11, 36, 12, 5, 37, 13),
        # 15
        (5, 109, 87, 1, 110, 88),
        (5, 65, 41, 5, 66, 42),
        (5, 54, 24, 7, 55, 25),
        (11, 36, 12, 7, 37, 13),
        # 16
        (5, 122, 98, 1, 123, 99),
        (7, 73, 45, 3, 74, 46),
        (15, 43, 19, 2, 44, 20),
        (3, 45, 15, 13, 46, 16),
        # 17
        (1, 135, 107, 5, 136, 108),
        (10, 74, 46, 1, 75, 47),
        (1, 50, 22, 15, 51, 23),
        (2, 42, 14, 17, 43, 15),
        # 18
        (5, 150, 120, 1, 151, 121),
        (9, 69, 43, 4, 70, 44),
        (17, 50, 22, 1, 51, 23),
        (2, 42, 14, 19, 43, 15),
        # 19
        (3, 141, 113, 4, 142, 114),
        (3, 70, 44, 11, 71, 45),
        (17, 47, 21, 4, 48, 22),
        (9, 39, 13, 16, 40, 14),
        # 20
        (3, 135, 107, 5, 136, 108),
        (3, 67, 41, 13, 68, 42),
        (15, 54, 24, 5, 55, 25),
        (15, 43, 15, 10, 44, 16),
        # 21
        (4, 144, 116, 4, 145, 117),
        (17, 68, 42),
        (17, 50, 22, 6, 51, 23),
        (19, 46, 16, 6, 47, 17),
        # 22
        (2, 139, 111, 7, 140, 112),
        (17, 74, 46),
        (7, 54, 24, 16, 55, 25),
        (34, 37, 13),
        # 23
        (4, 151, 121, 5, 152, 122),
        (4, 75, 47, 14, 76, 48),
        (11, 54, 24, 14, 55, 25),
        (16, 45, 15, 14, 46, 16),
        # 24
        (6, 147, 117, 4, 148, 118),
        (6, 73, 45, 14, 74, 46),
        (11, 54, 24, 16, 55, 25),
        (30, 46, 16, 2, 47, 17),
        # 25
        (8, 132, 106, 4, 133, 107),
        (8, 75, 47, 13, 76, 48),
        (7, 54, 24, 22, 55, 25),
        (22, 45, 15, 13, 46, 16),
        # 26
        (10, 142, 114, 2, 143, 115),
        (19, 74, 46, 4, 75, 47),
        (28, 50, 22, 6, 51, 23),
        (33, 46, 16, 4, 47, 17),
        # 27
        (8, 152, 122, 4, 153, 123),
        (22, 73, 45, 3, 74, 46),
        (8, 53, 23, 26, 54, 24),
        (12, 45, 15, 28, 46, 16),
        # 28
        (3, 147, 117, 10, 148, 118),
        (3, 73, 45, 23, 74, 46),
        (4, 54, 24, 31, 55, 25),
        (11, 45, 15, 31, 46, 16),
        # 29
        (7, 146, 116, 7, 147, 117),
        (21, 73, 45, 7, 74, 46),
        (1, 53, 23, 37, 54, 24),
        (19, 45, 15, 26, 46, 16),
        # 30
        (5, 145, 115, 10, 146, 116),
        (19, 75, 47, 10, 76, 48),
        (15, 54, 24, 25, 55, 25),
        (23, 45, 15, 25, 46, 16),
        # 31
        (13, 145, 115, 3, 146, 116),
        (2, 74, 46, 29, 75, 47),
        (42, 54, 24, 1, 55, 25),
        (23, 45, 15, 28, 46, 16),
        # 32
        (17, 145, 115),
        (10, 74, 46, 23, 75, 47),
        (10, 54, 24, 35, 55, 25),
        (19, 45, 15, 35, 46, 16),
        # 33
        (17, 145, 115, 1, 146, 116),
        (14, 74, 46, 21, 75, 47),
        (29, 54, 24, 19, 55, 25),
        (11, 45, 15, 46, 46, 16),
        # 34
        (13, 145, 115, 6, 146, 116),
        (14, 74, 46, 23, 75, 47),
        (44, 54, 24, 7, 55, 25),
        (59, 46, 16, 1, 47, 17),
        # 35
        (12, 151, 121, 7, 152, 122),
        (12, 75, 47, 26, 76, 48),
        (39, 54, 24, 14, 55, 25),
        (22, 45, 15, 41, 46, 16),
        # 36
        (6, 151, 121, 14, 152, 122),
        (6, 75, 47, 34, 76, 48),
        (46, 54, 24, 10, 55, 25),
        (2, 45, 15, 64, 46, 16),
        # 37
        (17, 152, 122, 4, 153, 123),
        (29, 74, 46, 14, 75, 47),
        (49, 54, 24, 10, 55, 25),
        (24, 45, 15, 46, 46, 16),
        # 38
        (4, 152, 122, 18, 153, 123),
        (13, 74, 46, 32, 75, 47),
        (48, 54, 24, 14, 55, 25),
        (42, 45, 15, 32, 46, 16),
        # 39
        (20, 147, 117, 4, 148, 118),
        (40, 75, 47, 7, 76, 48),
        (43, 54, 24, 22, 55, 25),
        (10, 45, 15, 67, 46, 16),
        # 40
        (19, 148, 118, 6, 149, 119),
        (18, 75, 47, 31, 76, 48),
        (34, 54, 24, 34, 55, 25),
        (20, 45, 15, 61, 46, 16),
    )


    def glog(n):
        if n < 1:  # pragma: no cover
            raise ValueError(f"glog({n})")
        return LOG_TABLE[n]


    def gexp(n):
        return EXP_TABLE[n % 255]


    class Polynomial:
        def __init__(self, num, shift):
            if not num:  # pragma: no cover
                raise Exception(f"{len(num)}/{shift}")

            offset = 0
            for offset in range(len(num)):
                if num[offset] != 0:
                    break

            self.num = num[offset:] + [0] * shift

        def __getitem__(self, index):
            return self.num[index]

        def __iter__(self):
            return iter(self.num)

        def __len__(self):
            return len(self.num)

        def __mul__(self, other):
            num = [0] * (len(self) + len(other) - 1)

            for i, item in enumerate(self):
                for j, other_item in enumerate(other):
                    num[i + j] ^= gexp(glog(item) + glog(other_item))

            return Polynomial(num, 0)

        def __mod__(self, other):
            difference = len(self) - len(other)
            if difference < 0:
                return self

            ratio = glog(self[0]) - glog(other[0])

            num = [
                item ^ gexp(glog(other_item) + ratio)
                for item, other_item in zip(self, other)
            ]
            if difference:
                num.extend(self[-difference:])

            # recursive call
            return Polynomial(num, 0) % other


    class RSBlock(NamedTuple):
        total_count: int
        data_count: int


    def rs_blocks1(version, error_correction):
        if error_correction not in RS_BLOCK_OFFSET:  # pragma: no cover
            raise Exception(
                "bad rs block @ version: %s / error_correction: %s"
                % (version, error_correction)
            )
        offset = RS_BLOCK_OFFSET[error_correction]
        rs_block = RS_BLOCK_TABLE[(version - 1) * 4 + offset]

        blocks = []

        for i in range(0, len(rs_block), 3):
            count, total_count, data_count = rs_block[i : i + 3]
            for _ in range(count):
                blocks.append(RSBlock(total_count, data_count))

        return blocks


    # QR encoding modes.
    MODE_NUMBER = 1 << 0
    MODE_ALPHA_NUM = 1 << 1
    MODE_8BIT_BYTE = 1 << 2
    MODE_KANJI = 1 << 3

    # Encoding mode sizes.
    MODE_SIZE_SMALL = {
        MODE_NUMBER: 10,
        MODE_ALPHA_NUM: 9,
        MODE_8BIT_BYTE: 8,
        MODE_KANJI: 8,
    }
    MODE_SIZE_MEDIUM = {
        MODE_NUMBER: 12,
        MODE_ALPHA_NUM: 11,
        MODE_8BIT_BYTE: 16,
        MODE_KANJI: 10,
    }
    MODE_SIZE_LARGE = {
        MODE_NUMBER: 14,
        MODE_ALPHA_NUM: 13,
        MODE_8BIT_BYTE: 16,
        MODE_KANJI: 12,
    }

    ALPHA_NUM = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    RE_ALPHA_NUM = re.compile(b"^[" + re.escape(ALPHA_NUM) + rb"]*\Z")

    # The number of bits for numeric delimited data lengths.
    NUMBER_LENGTH = {3: 10, 2: 7, 1: 4}

    PATTERN_POSITION_TABLE = [
        [],
        [6, 18],
        [6, 22],
        [6, 26],
        [6, 30],
        [6, 34],
        [6, 22, 38],
        [6, 24, 42],
        [6, 26, 46],
        [6, 28, 50],
        [6, 30, 54],
        [6, 32, 58],
        [6, 34, 62],
        [6, 26, 46, 66],
        [6, 26, 48, 70],
        [6, 26, 50, 74],
        [6, 30, 54, 78],
        [6, 30, 56, 82],
        [6, 30, 58, 86],
        [6, 34, 62, 90],
        [6, 28, 50, 72, 94],
        [6, 26, 50, 74, 98],
        [6, 30, 54, 78, 102],
        [6, 28, 54, 80, 106],
        [6, 32, 58, 84, 110],
        [6, 30, 58, 86, 114],
        [6, 34, 62, 90, 118],
        [6, 26, 50, 74, 98, 122],
        [6, 30, 54, 78, 102, 126],
        [6, 26, 52, 78, 104, 130],
        [6, 30, 56, 82, 108, 134],
        [6, 34, 60, 86, 112, 138],
        [6, 30, 58, 86, 114, 142],
        [6, 34, 62, 90, 118, 146],
        [6, 30, 54, 78, 102, 126, 150],
        [6, 24, 50, 76, 102, 128, 154],
        [6, 28, 54, 80, 106, 132, 158],
        [6, 32, 58, 84, 110, 136, 162],
        [6, 26, 54, 82, 110, 138, 166],
        [6, 30, 58, 86, 114, 142, 170],
    ]

    rsPoly_LUT = {
        7: [1, 127, 122, 154, 164, 11, 68, 117],
        10: [1, 216, 194, 159, 111, 199, 94, 95, 113, 157, 193],
        13: [1, 137, 73, 227, 17, 177, 17, 52, 13, 46, 43, 83, 132, 120],
        15: [1, 29, 196, 111, 163, 112, 74, 10, 105, 105, 139, 132, 151, 32, 134, 26],
        16: [1, 59, 13, 104, 189, 68, 209, 30, 8, 163, 65, 41, 229, 98, 50, 36, 59],
        17: [1, 119, 66, 83, 120, 119, 22, 197, 83, 249, 41, 143, 134, 85, 53, 125, 99, 79],
        18: [
            1,
            239,
            251,
            183,
            113,
            149,
            175,
            199,
            215,
            240,
            220,
            73,
            82,
            173,
            75,
            32,
            67,
            217,
            146,
        ],
        20: [
            1,
            152,
            185,
            240,
            5,
            111,
            99,
            6,
            220,
            112,
            150,
            69,
            36,
            187,
            22,
            228,
            198,
            121,
            121,
            165,
            174,
        ],
        22: [
            1,
            89,
            179,
            131,
            176,
            182,
            244,
            19,
            189,
            69,
            40,
            28,
            137,
            29,
            123,
            67,
            253,
            86,
            218,
            230,
            26,
            145,
            245,
        ],
        24: [
            1,
            122,
            118,
            169,
            70,
            178,
            237,
            216,
            102,
            115,
            150,
            229,
            73,
            130,
            72,
            61,
            43,
            206,
            1,
            237,
            247,
            127,
            217,
            144,
            117,
        ],
        26: [
            1,
            246,
            51,
            183,
            4,
            136,
            98,
            199,
            152,
            77,
            56,
            206,
            24,
            145,
            40,
            209,
            117,
            233,
            42,
            135,
            68,
            70,
            144,
            146,
            77,
            43,
            94,
        ],
        28: [
            1,
            252,
            9,
            28,
            13,
            18,
            251,
            208,
            150,
            103,
            174,
            100,
            41,
            167,
            12,
            247,
            56,
            117,
            119,
            233,
            127,
            181,
            100,
            121,
            147,
            176,
            74,
            58,
            197,
        ],
        30: [
            1,
            212,
            246,
            77,
            73,
            195,
            192,
            75,
            98,
            5,
            70,
            103,
            177,
            22,
            217,
            138,
            51,
            181,
            246,
            72,
            25,
            18,
            46,
            228,
            74,
            216,
            195,
            11,
            106,
            130,
            150,
        ],
    }


    G15 = (1 << 10) | (1 << 8) | (1 << 5) | (1 << 4) | (1 << 2) | (1 << 1) | (1 << 0)
    G18 = (
        (1 << 12)
        | (1 << 11)
        | (1 << 10)
        | (1 << 9)
        | (1 << 8)
        | (1 << 5)
        | (1 << 2)
        | (1 << 0)
    )
    G15_MASK = (1 << 14) | (1 << 12) | (1 << 10) | (1 << 4) | (1 << 1)

    PAD0 = 0xEC
    PAD1 = 0x11


    # Precompute bit count limits, indexed by error correction level and code size
    def _data_count(block):
        return block.data_count


    BIT_LIMIT_TABLE = [
        [0]
        + [
            8 * sum(map(_data_count, rs_blocks1(version, error_correction)))
            for version in range(1, 41)
        ]
        for error_correction in range(4)
    ]


    def BCH_type_info(data):
        d = data << 10
        while BCH_digit(d) - BCH_digit(G15) >= 0:
            d ^= G15 << (BCH_digit(d) - BCH_digit(G15))

        return ((data << 10) | d) ^ G15_MASK


    def BCH_type_number(data):
        d = data << 12
        while BCH_digit(d) - BCH_digit(G18) >= 0:
            d ^= G18 << (BCH_digit(d) - BCH_digit(G18))
        return (data << 12) | d


    def BCH_digit(data):
        digit = 0
        while data != 0:
            digit += 1
            data >>= 1
        return digit


    def pattern_position(version):
        return PATTERN_POSITION_TABLE[version - 1]


    def mask_func(pattern):
        """
        Return the mask function for the given mask pattern.
        """
        if pattern == 0:  # 000
            return lambda i, j: (i + j) % 2 == 0
        if pattern == 1:  # 001
            return lambda i, j: i % 2 == 0
        if pattern == 2:  # 010
            return lambda i, j: j % 3 == 0
        if pattern == 3:  # 011
            return lambda i, j: (i + j) % 3 == 0
        if pattern == 4:  # 100
            return lambda i, j: (math.floor(i / 2) + math.floor(j / 3)) % 2 == 0
        if pattern == 5:  # 101
            return lambda i, j: (i * j) % 2 + (i * j) % 3 == 0
        if pattern == 6:  # 110
            return lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0
        if pattern == 7:  # 111
            return lambda i, j: ((i * j) % 3 + (i + j) % 2) % 2 == 0
        raise TypeError("Bad mask pattern: " + pattern)  # pragma: no cover


    def mode_sizes_for_version(version):
        if version < 10:
            return MODE_SIZE_SMALL
        elif version < 27:
            return MODE_SIZE_MEDIUM
        else:
            return MODE_SIZE_LARGE


    def length_in_bits(mode, version):
        if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE, MODE_KANJI):
            raise TypeError(f"Invalid mode ({mode})")  # pragma: no cover

        check_version(version)

        return mode_sizes_for_version(version)[mode]


    def check_version(version):
        if version < 1 or version > 40:
            raise ValueError(f"Invalid version (was {version}, expected 1 to 40)")


    def lost_point(modules):
        modules_count = len(modules)

        lost_point = 0

        lost_point = _lost_point_level1(modules, modules_count)
        lost_point += _lost_point_level2(modules, modules_count)
        lost_point += _lost_point_level3(modules, modules_count)
        lost_point += _lost_point_level4(modules, modules_count)

        return lost_point


    def _lost_point_level1(modules, modules_count):
        lost_point = 0

        modules_range = range(modules_count)
        container = [0] * (modules_count + 1)

        for row in modules_range:
            this_row = modules[row]
            previous_color = this_row[0]
            length = 0
            for col in modules_range:
                if this_row[col] == previous_color:
                    length += 1
                else:
                    if length >= 5:
                        container[length] += 1
                    length = 1
                    previous_color = this_row[col]
            if length >= 5:
                container[length] += 1

        for col in modules_range:
            previous_color = modules[0][col]
            length = 0
            for row in modules_range:
                if modules[row][col] == previous_color:
                    length += 1
                else:
                    if length >= 5:
                        container[length] += 1
                    length = 1
                    previous_color = modules[row][col]
            if length >= 5:
                container[length] += 1

        lost_point += sum(
            container[each_length] * (each_length - 2)
            for each_length in range(5, modules_count + 1)
        )

        return lost_point


    def _lost_point_level2(modules, modules_count):
        lost_point = 0

        modules_range = range(modules_count - 1)
        for row in modules_range:
            this_row = modules[row]
            next_row = modules[row + 1]
            # use iter() and next() to skip next four-block. e.g.
            # d a f   if top-right a != b bottom-right,
            # c b e   then both abcd and abef won't lost any point.
            modules_range_iter = iter(modules_range)
            for col in modules_range_iter:
                top_right = this_row[col + 1]
                if top_right != next_row[col + 1]:
                    # reduce 33.3% of runtime via next().
                    # None: raise nothing if there is no next item.
                    next(modules_range_iter, None)
                elif top_right != this_row[col]:
                    continue
                elif top_right != next_row[col]:
                    continue
                else:
                    lost_point += 3

        return lost_point


    def _lost_point_level3(modules, modules_count):
        # 1 : 1 : 3 : 1 : 1 ratio (dark:light:dark:light:dark) pattern in
        # row/column, preceded or followed by light area 4 modules wide. From ISOIEC.
        # pattern1:     10111010000
        # pattern2: 00001011101
        modules_range = range(modules_count)
        modules_range_short = range(modules_count - 10)
        lost_point = 0

        for row in modules_range:
            this_row = modules[row]
            modules_range_short_iter = iter(modules_range_short)
            col = 0
            for col in modules_range_short_iter:
                if (
                    not this_row[col + 1]
                    and this_row[col + 4]
                    and not this_row[col + 5]
                    and this_row[col + 6]
                    and not this_row[col + 9]
                    and (
                        this_row[col + 0]
                        and this_row[col + 2]
                        and this_row[col + 3]
                        and not this_row[col + 7]
                        and not this_row[col + 8]
                        and not this_row[col + 10]
                        or not this_row[col + 0]
                        and not this_row[col + 2]
                        and not this_row[col + 3]
                        and this_row[col + 7]
                        and this_row[col + 8]
                        and this_row[col + 10]
                    )
                ):
                    lost_point += 40
                # horspool algorithm.
                # if this_row[col + 10]:
                #   pattern1 shift 4, pattern2 shift 2. So min=2.
                # else:
                #   pattern1 shift 1, pattern2 shift 1. So min=1.
                if this_row[col + 10]:
                    next(modules_range_short_iter, None)

        for col in modules_range:
            modules_range_short_iter = iter(modules_range_short)
            row = 0
            for row in modules_range_short_iter:
                if (
                    not modules[row + 1][col]
                    and modules[row + 4][col]
                    and not modules[row + 5][col]
                    and modules[row + 6][col]
                    and not modules[row + 9][col]
                    and (
                        modules[row + 0][col]
                        and modules[row + 2][col]
                        and modules[row + 3][col]
                        and not modules[row + 7][col]
                        and not modules[row + 8][col]
                        and not modules[row + 10][col]
                        or not modules[row + 0][col]
                        and not modules[row + 2][col]
                        and not modules[row + 3][col]
                        and modules[row + 7][col]
                        and modules[row + 8][col]
                        and modules[row + 10][col]
                    )
                ):
                    lost_point += 40
                if modules[row + 10][col]:
                    next(modules_range_short_iter, None)

        return lost_point


    def _lost_point_level4(modules, modules_count):
        dark_count = sum(map(sum, modules))
        percent = float(dark_count) / (modules_count**2)
        # Every 5% departure from 50%, rating++
        rating = int(abs(percent * 100 - 50) / 5)
        return rating * 10


    def optimal_data_chunks(data, minimum=4):
        """
        An iterator returning QRData chunks optimized to the data content.

        :param minimum: The minimum number of bytes in a row to split as a chunk.
        """
        data = to_bytestring(data)
        num_pattern = rb"\d"
        alpha_pattern = b"[" + re.escape(ALPHA_NUM) + b"]"
        if len(data) <= minimum:
            num_pattern = re.compile(b"^" + num_pattern + b"+$")
            alpha_pattern = re.compile(b"^" + alpha_pattern + b"+$")
        else:
            re_repeat = b"{" + str(minimum).encode("ascii") + b",}"
            num_pattern = re.compile(num_pattern + re_repeat)
            alpha_pattern = re.compile(alpha_pattern + re_repeat)
        num_bits = _optimal_split(data, num_pattern)
        for is_num, chunk in num_bits:
            if is_num:
                yield QRData(chunk, mode=MODE_NUMBER, check_data=False)
            else:
                for is_alpha, sub_chunk in _optimal_split(chunk, alpha_pattern):
                    mode = MODE_ALPHA_NUM if is_alpha else MODE_8BIT_BYTE
                    yield QRData(sub_chunk, mode=mode, check_data=False)


    def _optimal_split(data, pattern):
        while data:
            match = re.search(pattern, data)
            if not match:
                break
            start, end = match.start(), match.end()
            if start:
                yield False, data[:start]
            yield True, data[start:end]
            data = data[end:]
        if data:
            yield False, data


    def to_bytestring(data):
        """
        Convert data to a (utf-8 encoded) byte-string if it isn't a byte-string
        already.
        """
        if not isinstance(data, bytes):
            data = str(data).encode("utf-8")
        return data


    def optimal_mode(data):
        """
        Calculate the optimal mode for this chunk of data.
        """
        if data.isdigit():
            return MODE_NUMBER
        if RE_ALPHA_NUM.match(data):
            return MODE_ALPHA_NUM
        return MODE_8BIT_BYTE


    class QRData:
        """
        Data held in a QR compatible format.

        Doesn't currently handle KANJI.
        """

        def __init__(self, data, mode=None, check_data=True):
            """
            If ``mode`` isn't provided, the most compact QR data type possible is
            chosen.
            """
            if check_data:
                data = to_bytestring(data)

            if mode is None:
                self.mode = optimal_mode(data)
            else:
                self.mode = mode
                if mode not in (MODE_NUMBER, MODE_ALPHA_NUM, MODE_8BIT_BYTE):
                    raise TypeError(f"Invalid mode ({mode})")  # pragma: no cover
                if check_data and mode < optimal_mode(data):  # pragma: no cover
                    raise ValueError(f"Provided data can not be represented in mode {mode}")

            self.data = data

        def __len__(self):
            return len(self.data)

        def write(self, buffer):
            if self.mode == MODE_NUMBER:
                for i in range(0, len(self.data), 3):
                    chars = self.data[i : i + 3]
                    bit_length = NUMBER_LENGTH[len(chars)]
                    buffer.put(int(chars), bit_length)
            elif self.mode == MODE_ALPHA_NUM:
                for i in range(0, len(self.data), 2):
                    chars = self.data[i : i + 2]
                    if len(chars) > 1:
                        buffer.put(
                            ALPHA_NUM.find(chars[0]) * 45 + ALPHA_NUM.find(chars[1]), 11
                        )
                    else:
                        buffer.put(ALPHA_NUM.find(chars), 6)
            else:
                # Iterating a bytestring in Python 3 returns an integer,
                # no need to ord().
                data = self.data
                for c in data:
                    buffer.put(c, 8)

        def __repr__(self):
            return repr(self.data)


    class BitBuffer:
        def __init__(self):
            self.buffer: List[int] = []
            self.length = 0

        def __repr__(self):
            return ".".join([str(n) for n in self.buffer])

        def get(self, index):
            buf_index = math.floor(index / 8)
            return ((self.buffer[buf_index] >> (7 - index % 8)) & 1) == 1

        def put(self, num, length):
            for i in range(length):
                self.put_bit(((num >> (length - i - 1)) & 1) == 1)

        def __len__(self):
            return self.length

        def put_bit(self, bit):
            buf_index = self.length // 8
            if len(self.buffer) <= buf_index:
                self.buffer.append(0)
            if bit:
                self.buffer[buf_index] |= 0x80 >> (self.length % 8)
            self.length += 1


    def create_bytes(buffer: BitBuffer, rs_blocks: List[RSBlock]):
        offset = 0

        maxDcCount = 0
        maxEcCount = 0

        dcdata: List[List[int]] = []
        ecdata: List[List[int]] = []

        for rs_block in rs_blocks:
            dcCount = rs_block.data_count
            ecCount = rs_block.total_count - dcCount

            maxDcCount = max(maxDcCount, dcCount)
            maxEcCount = max(maxEcCount, ecCount)

            current_dc = [0xFF & buffer.buffer[i + offset] for i in range(dcCount)]
            offset += dcCount

            # Get error correction polynomial.
            if ecCount in rsPoly_LUT:
                rsPoly = Polynomial(rsPoly_LUT[ecCount], 0)
            else:
                rsPoly = Polynomial([1], 0)
                for i in range(ecCount):
                    rsPoly = rsPoly * Polynomial([1, gexp(i)], 0)

            rawPoly = Polynomial(current_dc, len(rsPoly) - 1)

            modPoly = rawPoly % rsPoly
            current_ec = []
            mod_offset = len(modPoly) - ecCount
            for i in range(ecCount):
                modIndex = i + mod_offset
                current_ec.append(modPoly[modIndex] if (modIndex >= 0) else 0)

            dcdata.append(current_dc)
            ecdata.append(current_ec)

        data = []
        for i in range(maxDcCount):
            for dc in dcdata:
                if i < len(dc):
                    data.append(dc[i])
        for i in range(maxEcCount):
            for ec in ecdata:
                if i < len(ec):
                    data.append(ec[i])

        return data


    def create_data(version, error_correction, data_list):

        buffer = BitBuffer()
        for data in data_list:
            buffer.put(data.mode, 4)
            buffer.put(len(data), length_in_bits(data.mode, version))
            data.write(buffer)

        # Calculate the maximum number of bits for the given version.
        rs_blocks = rs_blocks1(version, error_correction)
        bit_limit = sum(block.data_count * 8 for block in rs_blocks)
        if len(buffer) > bit_limit:
            raise DataOverflowError(
                "Code length overflow. Data size (%s) > size available (%s)"
                % (len(buffer), bit_limit)
            )

        # Terminate the bits (add up to four 0s).
        for _ in range(min(bit_limit - len(buffer), 4)):
            buffer.put_bit(False)

        # Delimit the string into 8-bit words, padding with 0s if necessary.
        delimit = len(buffer) % 8
        if delimit:
            for _ in range(8 - delimit):
                buffer.put_bit(False)

        # Add special alternating padding bitstrings until buffer is full.
        bytes_to_fill = (bit_limit - len(buffer)) // 8
        for i in range(bytes_to_fill):
            if i % 2 == 0:
                buffer.put(PAD0, 8)
            else:
                buffer.put(PAD1, 8)

        return create_bytes(buffer, rs_blocks)


    class DataOverflowError(Exception):
        pass


    class QRModuleDrawer(abc.ABC):
        needs_neighbors = False
        def __init__(self, **kwargs):
            pass
        def initialize(self, img: "BaseImage") -> None:
            self.img = img
        @abc.abstractmethod
        def drawrect(self, box, is_active) -> None:
            ...


    class Error(Exception):
        def __str__(self):
            return self.__class__.__name__ + ': ' + ' '.join(self.args)


    class FormatError(Error):
        """
        Problem with input file format.
        In other words, PNG file does not conform to
        the specification in some way and is invalid.
        """


    class ProtocolError(Error):
        """
        Problem with the way the programming interface has been used,
        or the data presented to it.
        """


    class ChunkError(FormatError):
        pass


    class Default:
        """The default for the greyscale parameter."""


    class Writer:
        """
        PNG encoder in pure Python.
        """

        def __init__(self, width=None, height=None,
                    size=None,
                    greyscale=Default,
                    alpha=False,
                    bitdepth=8,
                    palette=None,
                    transparent=None,
                    background=None,
                    gamma=None,
                    compression=None,
                    interlace=False,
                    planes=None,
                    colormap=None,
                    maxval=None,
                    chunk_limit=2**20,
                    x_pixels_per_unit=None,
                    y_pixels_per_unit=None,
                    unit_is_meter=False):

            width, height = check_sizes(size, width, height)
            del size

            if not is_natural(width) or not is_natural(height):
                raise ProtocolError("width and height must be integers")
            if width <= 0 or height <= 0:
                raise ProtocolError("width and height must be greater than zero")
            # http://www.w3.org/TR/PNG/#7Integers-and-byte-order
            if width > 2 ** 31 - 1 or height > 2 ** 31 - 1:
                raise ProtocolError("width and height cannot exceed 2**31-1")

            if alpha and transparent is not None:
                raise ProtocolError(
                    "transparent colour not allowed with alpha channel")

            # bitdepth is either single integer, or tuple of integers.
            # Convert to tuple.
            try:
                len(bitdepth)
            except TypeError:
                bitdepth = (bitdepth, )
            for b in bitdepth:
                valid = is_natural(b) and 1 <= b <= 16
                if not valid:
                    raise ProtocolError(
                        "each bitdepth %r must be a positive integer <= 16" %
                        (bitdepth,))

            # Calculate channels, and
            # expand bitdepth to be one element per channel.
            palette = check_palette(palette)
            alpha = bool(alpha)
            colormap = bool(palette)
            if greyscale is Default and palette:
                greyscale = False
            greyscale = bool(greyscale)
            if colormap:
                color_planes = 1
                planes = 1
            else:
                color_planes = (3, 1)[greyscale]
                planes = color_planes + alpha
            if len(bitdepth) == 1:
                bitdepth *= planes

            bitdepth, self.rescale = check_bitdepth_rescale(
                    palette,
                    bitdepth,
                    transparent, alpha, greyscale)

            # These are assertions, because above logic should have
            # corrected or raised all problematic cases.
            if bitdepth < 8:
                assert greyscale or palette
                assert not alpha
            if bitdepth > 8:
                assert not palette

            transparent = check_color(transparent, greyscale, 'transparent')
            background = check_color(background, greyscale, 'background')

            # It's important that the true boolean values
            # (greyscale, alpha, colormap, interlace) are converted
            # to bool because Iverson's convention is relied upon later on.
            self.width = width
            self.height = height
            self.transparent = transparent
            self.background = background
            self.gamma = gamma
            self.greyscale = greyscale
            self.alpha = alpha
            self.colormap = colormap
            self.bitdepth = int(bitdepth)
            self.compression = compression
            self.chunk_limit = chunk_limit
            self.interlace = bool(interlace)
            self.palette = palette
            self.x_pixels_per_unit = x_pixels_per_unit
            self.y_pixels_per_unit = y_pixels_per_unit
            self.unit_is_meter = bool(unit_is_meter)

            self.color_type = (4 * self.alpha +
                            2 * (not greyscale) +
                            1 * self.colormap)
            assert self.color_type in (0, 2, 3, 4, 6)

            self.color_planes = color_planes
            self.planes = planes
            # :todo: fix for bitdepth < 8
            self.psize = (self.bitdepth / 8) * self.planes

        def write(self, outfile, rows):
            """
            Write a PNG image to the output file.
            `rows` should be an iterable that yields each row
            (each row is a sequence of values).
            The rows should be the rows of the original image,
            so there should be ``self.height`` rows of
            ``self.width * self.planes`` values.
            If `interlace` is specified (when creating the instance),
            then an interlaced PNG file will be written.
            Supply the rows in the normal image order;
            the interlacing is carried out internally.

            .. note ::

            Interlacing requires the entire image to be in working memory.
            """

            # Values per row
            vpr = self.width * self.planes

            def check_rows(rows):
                """
                Yield each row in rows,
                but check each row first (for correct width).
                """
                for i, row in enumerate(rows):
                    try:
                        wrong_length = len(row) != vpr
                    except TypeError:
                        # When using an itertools.ichain object or
                        # other generator not supporting __len__,
                        # we set this to False to skip the check.
                        wrong_length = False
                    if wrong_length:
                        # Note: row numbers start at 0.
                        raise ProtocolError(
                            "Expected %d values but got %d values, in row %d" %
                            (vpr, len(row), i))
                    yield row

            if self.interlace:
                fmt = 'BH'[self.bitdepth > 8]
                a = array(fmt, itertools.chain(*check_rows(rows)))
                return self.write_array(outfile, a)

            nrows = self.write_passes(outfile, check_rows(rows))
            if nrows != self.height:
                raise ProtocolError(
                    "rows supplied (%d) does not match height (%d)" %
                    (nrows, self.height))
            return nrows

        def write_passes(self, outfile, rows):
            """
            Write a PNG image to the output file.

            Most users are expected to find the :meth:`write` or
            :meth:`write_array` method more convenient.

            The rows should be given to this method in the order that
            they appear in the output file.
            For straightlaced images, this is the usual top to bottom ordering.
            For interlaced images the rows should have been interlaced before
            passing them to this function.

            `rows` should be an iterable that yields each row
            (each row being a sequence of values).
            """

            # Ensure rows are scaled (to 4-/8-/16-bit),
            # and packed into bytes.

            if self.rescale:
                rows = rescale_rows(rows, self.rescale)

            if self.bitdepth < 8:
                rows = pack_rows(rows, self.bitdepth)
            elif self.bitdepth == 16:
                rows = unpack_rows(rows)

            return self.write_packed(outfile, rows)

        def write_packed(self, outfile, rows):
            """
            Write PNG file to `outfile`.
            `rows` should be an iterator that yields each packed row;
            a packed row being a sequence of packed bytes.

            The rows have a filter byte prefixed and
            are then compressed into one or more IDAT chunks.
            They are not processed any further,
            so if bitdepth is other than 1, 2, 4, 8, 16,
            the pixel values should have been scaled
            before passing them to this method.

            This method does work for interlaced images but it is best avoided.
            For interlaced images, the rows should be
            presented in the order that they appear in the file.
            """

            self.write_preamble(outfile)

            # http://www.w3.org/TR/PNG/#11IDAT
            if self.compression is not None:
                compressor = zlib.compressobj(self.compression)
            else:
                compressor = zlib.compressobj()

            # data accumulates bytes to be compressed for the IDAT chunk;
            # it's compressed when sufficiently large.
            data = bytearray()

            # raise i scope out of the for loop. set to -1, because the for loop
            # sets i to 0 on the first pass
            i = -1
            for i, row in enumerate(rows):
                # Add "None" filter type.
                # Currently, it's essential that this filter type be used
                # for every scanline as
                # we do not mark the first row of a reduced pass image;
                # that means we could accidentally compute
                # the wrong filtered scanline if we used
                # "up", "average", or "paeth" on such a line.
                data.append(0)
                data.extend(row)
                if len(data) > self.chunk_limit:
                    compressed = compressor.compress(data)
                    if len(compressed):
                        write_chunk(outfile, b'IDAT', compressed)
                    data = bytearray()

            compressed = compressor.compress(bytes(data))
            flushed = compressor.flush()
            if len(compressed) or len(flushed):
                write_chunk(outfile, b'IDAT', compressed + flushed)
            # http://www.w3.org/TR/PNG/#11IEND
            write_chunk(outfile, b'IEND')
            return i + 1

        def write_preamble(self, outfile):
            # http://www.w3.org/TR/PNG/#5PNG-file-signature

            # This is the first write that is made when
            # writing a PNG file.
            # This one, and only this one, is checked for TypeError,
            # which generally indicates that we are writing bytes
            # into a text stream.
            try:
                outfile.write(signature)
            except TypeError as e:
                raise ProtocolError("PNG must be written to a binary stream") from e

            # http://www.w3.org/TR/PNG/#11IHDR
            write_chunk(outfile, b'IHDR',
                        struct.pack("!2I5B", self.width, self.height,
                                    self.bitdepth, self.color_type,
                                    0, 0, self.interlace))

            # See :chunk:order
            # http://www.w3.org/TR/PNG/#11gAMA
            if self.gamma is not None:
                write_chunk(outfile, b'gAMA',
                            struct.pack("!L", int(round(self.gamma * 1e5))))

            # See :chunk:order
            # http://www.w3.org/TR/PNG/#11sBIT
            if self.rescale:
                write_chunk(
                    outfile, b'sBIT',
                    struct.pack('%dB' % self.planes,
                                * [s[0] for s in self.rescale]))

            # :chunk:order: Without a palette (PLTE chunk),
            # ordering is relatively relaxed.
            # With one, gAMA chunk must precede PLTE chunk
            # which must precede tRNS and bKGD.
            # See http://www.w3.org/TR/PNG/#5ChunkOrdering
            if self.palette:
                p, t = make_palette_chunks(self.palette)
                write_chunk(outfile, b'PLTE', p)
                if t:
                    # tRNS chunk is optional;
                    # Only needed if palette entries have alpha.
                    write_chunk(outfile, b'tRNS', t)

            # http://www.w3.org/TR/PNG/#11tRNS
            if self.transparent is not None:
                if self.greyscale:
                    fmt = "!1H"
                else:
                    fmt = "!3H"
                write_chunk(outfile, b'tRNS',
                            struct.pack(fmt, *self.transparent))

            # http://www.w3.org/TR/PNG/#11bKGD
            if self.background is not None:
                if self.greyscale:
                    fmt = "!1H"
                else:
                    fmt = "!3H"
                write_chunk(outfile, b'bKGD',
                            struct.pack(fmt, *self.background))

            # http://www.w3.org/TR/PNG/#11pHYs
            if (self.x_pixels_per_unit is not None and
                    self.y_pixels_per_unit is not None):
                tup = (self.x_pixels_per_unit,
                    self.y_pixels_per_unit,
                    int(self.unit_is_meter))
                write_chunk(outfile, b'pHYs', struct.pack("!LLB", *tup))

        def write_array(self, outfile, pixels):
            """
            Write an array that holds all the image values
            as a PNG file on the output file.
            See also :meth:`write` method.
            """

            if self.interlace:
                if type(pixels) != array:
                    # Coerce to array type
                    fmt = 'BH'[self.bitdepth > 8]
                    pixels = array(fmt, pixels)
                return self.write_passes(
                    outfile,
                    self.array_scanlines_interlace(pixels)
                )
            else:
                return self.write_passes(
                    outfile,
                    self.array_scanlines(pixels)
                )

        def array_scanlines(self, pixels):
            """
            Generates rows (each a sequence of values) from
            a single array of values.
            """

            # Values per row
            vpr = self.width * self.planes
            stop = 0
            for y in range(self.height):
                start = stop
                stop = start + vpr
                yield pixels[start:stop]

        def array_scanlines_interlace(self, pixels):
            """
            Generator for interlaced scanlines from an array.
            `pixels` is the full source image as a single array of values.
            The generator yields each scanline of the reduced passes in turn,
            each scanline being a sequence of values.
            """

            # http://www.w3.org/TR/PNG/#8InterlaceMethods
            # Array type.
            fmt = 'BH'[self.bitdepth > 8]
            # Value per row
            vpr = self.width * self.planes

            # Each iteration generates a scanline starting at (x, y)
            # and consisting of every xstep pixels.
            for lines in adam7_generate(self.width, self.height):
                for x, y, xstep in lines:
                    # Pixels per row (of reduced image)
                    ppr = int(math.ceil((self.width - x) / float(xstep)))
                    # Values per row (of reduced image)
                    reduced_row_len = ppr * self.planes
                    if xstep == 1:
                        # Easy case: line is a simple slice.
                        offset = y * vpr
                        yield pixels[offset: offset + vpr]
                        continue
                    # We have to step by xstep,
                    # which we can do one plane at a time
                    # using the step in Python slices.
                    row = array(fmt)
                    # There's no easier way to set the length of an array
                    row.extend(pixels[0:reduced_row_len])
                    offset = y * vpr + x * self.planes
                    end_offset = (y + 1) * vpr
                    skip = self.planes * xstep
                    for i in range(self.planes):
                        row[i::self.planes] = \
                            pixels[offset + i: end_offset: skip]
                    yield row


    DrawerAliases = Dict[str, Tuple[Type[QRModuleDrawer], Dict[str, Any]]]
    ModulesType = List[List[Optional[bool]]]
    # Cache modules generated just based on the QR Code version
    precomputed_qr_blanks: Dict[int, ModulesType] = {}

    def copy_2d_array(x):
        return [row[:] for row in x]

    class ActiveWithNeighbors(NamedTuple):
        NW: bool
        N: bool
        NE: bool
        W: bool
        me: bool
        E: bool
        SW: bool
        S: bool
        SE: bool

        def __bool__(self) -> bool:
            return self.me


    class BaseImage:

        kind: Optional[str] = None
        allowed_kinds: Optional[Tuple[str]] = None
        needs_context = False
        needs_processing = False
        needs_drawrect = True

        def __init__(self, border, width, box_size, *args, **kwargs):
            self.border = border
            self.width = width
            self.box_size = box_size
            self.pixel_size = (self.width + self.border * 2) * self.box_size
            self.modules = kwargs.pop("qrcode_modules")
            self._img = self.new_image(**kwargs)
            self.init_new_image()

        @abc.abstractmethod
        def drawrect(self, row, col):
            """
            Draw a single rectangle of the QR code.
            """

        def drawrect_context(self, row: int, col: int, qr: "QRCode"):
            """
            Draw a single rectangle of the QR code given the surrounding context
            """
            raise NotImplementedError("BaseImage.drawrect_context")  # pragma: no cover

        def process(self):
            """
            Processes QR code after completion
            """
            raise NotImplementedError("BaseImage.drawimage")  # pragma: no cover

        @abc.abstractmethod
        def save(self, stream, kind=None):
            """
            Save the image file.
            """

        def pixel_box(self, row, col):
            """
            A helper method for pixel-based image generators that specifies the
            four pixel coordinates for a single rect.
            """
            x = (col + self.border) * self.box_size
            y = (row + self.border) * self.box_size
            return (
                (x, y),
                (x + self.box_size - 1, y + self.box_size - 1),
            )

        @abc.abstractmethod
        def new_image(self, **kwargs) -> Any:
            """
            Build the image class. Subclasses should return the class created.
            """

        def init_new_image(self):
            pass

        def get_image(self, **kwargs):
            """
            Return the image class for further processing.
            """
            return self._img

        def check_kind(self, kind, transform=None):
            """
            Get the image type.
            """
            if kind is None:
                kind = self.kind
            allowed = not self.allowed_kinds or kind in self.allowed_kinds
            if transform:
                kind = transform(kind)
                if not allowed:
                    allowed = kind in self.allowed_kinds
            if not allowed:
                raise ValueError(f"Cannot set {type(self).__name__} type to {kind}")
            return kind

        def is_eye(self, row: int, col: int):
            """
            Find whether the referenced module is in an eye.
            """
            return (
                (row < 7 and col < 7)
                or (row < 7 and self.width - col < 8)
                or (self.width - row < 8 and col < 7)
            )

    GenericImage = TypeVar("GenericImage", bound=BaseImage)
    GenericImageLocal = TypeVar("GenericImageLocal", bound=BaseImage)

    class BaseImageWithDrawer(BaseImage):
        default_drawer_class: Type[QRModuleDrawer]
        drawer_aliases: DrawerAliases = {}

        def get_default_module_drawer(self) -> QRModuleDrawer:
            return self.default_drawer_class()

        def get_default_eye_drawer(self) -> QRModuleDrawer:
            return self.default_drawer_class()

        needs_context = True

        module_drawer: "QRModuleDrawer"
        eye_drawer: "QRModuleDrawer"

        def __init__(
            self,
            *args,
            module_drawer: Union[QRModuleDrawer, str, None] = None,
            eye_drawer: Union[QRModuleDrawer, str, None] = None,
            **kwargs,
        ):
            self.module_drawer = (
                self.get_drawer(module_drawer) or self.get_default_module_drawer()
            )
            # The eye drawer can be overridden by another module drawer as well,
            # but you have to be more careful with these in order to make the QR
            # code still parseable
            self.eye_drawer = self.get_drawer(eye_drawer) or self.get_default_eye_drawer()
            super().__init__(*args, **kwargs)

        def get_drawer(
            self, drawer: Union[QRModuleDrawer, str, None]
        ) -> Optional[QRModuleDrawer]:
            if not isinstance(drawer, str):
                return drawer
            drawer_cls, kwargs = self.drawer_aliases[drawer]
            return drawer_cls(**kwargs)

        def init_new_image(self):
            self.module_drawer.initialize(img=self)
            self.eye_drawer.initialize(img=self)

            return super().init_new_image()

        def drawrect_context(self, row: int, col: int, qr: "QRCode"):
            box = self.pixel_box(row, col)
            drawer = self.eye_drawer if self.is_eye(row, col) else self.module_drawer
            is_active: Union[bool, ActiveWithNeighbors] = (
                qr.active_with_neighbors(row, col)
                if drawer.needs_neighbors
                else bool(qr.modules[row][col])
            )

            drawer.drawrect(box, is_active)


    class PyPNGImage(BaseImage):
        """
        pyPNG image builder.
        """

        kind = "PNG"
        allowed_kinds = ("PNG",)
        needs_drawrect = False

        def new_image(self, **kwargs):
            return Writer(self.pixel_size, self.pixel_size, greyscale=True, bitdepth=1)

        def drawrect(self, row, col):
            """
            Not used.
            """

        def save(self, stream, kind=None):
            if isinstance(stream, str):
                stream = open(stream, "wb")
            self._img.write(stream, self.rows_iter())

        def rows_iter(self):
            yield from self.border_rows_iter()
            border_col = [1] * (self.box_size * self.border)
            for module_row in self.modules:
                row = (
                    border_col
                    + list(
                        chain.from_iterable(
                            ([not point] * self.box_size) for point in module_row
                        )
                    )
                    + border_col
                )
                for _ in range(self.box_size):
                    yield row
            yield from self.border_rows_iter()

        def border_rows_iter(self):
            border_row = [1] * (self.box_size * (self.width + self.border * 2))
            for _ in range(self.border * self.box_size):
                yield border_row


    class PilImage(BaseImage):
        """
        PIL image builder, default format is PNG.
        """

        kind = "PNG"

        def new_image(self, **kwargs):
            back_color = kwargs.get("back_color", "white")
            fill_color = kwargs.get("fill_color", "black")

            try:
                fill_color = fill_color.lower()
            except AttributeError:
                pass

            try:
                back_color = back_color.lower()
            except AttributeError:
                pass

            # L mode (1 mode) color = (r*299 + g*587 + b*114)//1000
            if fill_color == "black" and back_color == "white":
                mode = "1"
                fill_color = 0
                if back_color == "white":
                    back_color = 255
            elif back_color == "transparent":
                mode = "RGBA"
                back_color = None
            else:
                mode = "RGB"

            img = Image.new(mode, (self.pixel_size, self.pixel_size), back_color)
            self.fill_color = fill_color
            self._idr = ImageDraw.Draw(img)
            return img

        def drawrect(self, row, col):
            box = self.pixel_box(row, col)
            self._idr.rectangle(box, fill=self.fill_color)

        def save(self, stream, format=None, **kwargs):
            kind = kwargs.pop("kind", self.kind)
            if format is None:
                format = kind
            self._img.save(stream, format=format, **kwargs)

        def __getattr__(self, name):
            return getattr(self._img, name)



    class QRCode(Generic[GenericImage]):
        modules: ModulesType
        _version: Optional[int] = None

        def __init__(
            self,
            version=None,
            error_correction=0,
            box_size=10,
            border=4,
            image_factory: Optional[Type[GenericImage]] = None,
            mask_pattern=None,
        ):
            self.version = version
            self.error_correction = int(error_correction)
            self.box_size = int(box_size)
            # Spec says border should be at least four boxes wide, but allow for
            # any (e.g. for producing printable QR codes).
            self.border = int(border)
            self.mask_pattern = mask_pattern
            self.image_factory = image_factory
            if image_factory is not None:
                assert issubclass(image_factory, BaseImage)
            self.clear()

        @property
        def version(self) -> int:
            if self._version is None:
                self.best_fit()
            return cast(int, self._version)

        @version.setter
        def version(self, value) -> None:
            if value is not None:
                value = int(value)
                check_version(value)
            self._version = value

        @property
        def mask_pattern(self):
            return self._mask_pattern

        @mask_pattern.setter
        def mask_pattern(self, pattern):
            self._mask_pattern = pattern

        def clear(self):
            """
            Reset the internal data.
            """
            self.modules = [[]]
            self.modules_count = 0
            self.data_cache = None
            self.data_list = []

        def add_data(self, data, optimize=20):
            """
            Add data to this QR Code.

            :param optimize: Data will be split into multiple chunks to optimize
                the QR size by finding to more compressed modes of at least this
                length. Set to ``0`` to avoid optimizing at all.
            """
            if isinstance(data, QRData):
                self.data_list.append(data)
            elif optimize:
                self.data_list.extend(optimal_data_chunks(data, minimum=optimize))
            else:
                self.data_list.append(QRData(data))
            self.data_cache = None

        def make(self, fit=True):
            """
            Compile the data into a QR Code array.

            :param fit: If ``True`` (or if a size has not been provided), find the
                best fit for the data to avoid data overflow errors.
            """
            if fit or (self.version is None):
                self.best_fit(start=self.version)
            if self.mask_pattern is None:
                self.makeImpl(False, self.best_mask_pattern())
            else:
                self.makeImpl(False, self.mask_pattern)

        def makeImpl(self, test, mask_pattern):
            self.modules_count = self.version * 4 + 17

            if self.version in precomputed_qr_blanks:
                self.modules = copy_2d_array(precomputed_qr_blanks[self.version])
            else:
                self.modules = [
                    [None] * self.modules_count for i in range(self.modules_count)
                ]
                self.setup_position_probe_pattern(0, 0)
                self.setup_position_probe_pattern(self.modules_count - 7, 0)
                self.setup_position_probe_pattern(0, self.modules_count - 7)
                self.setup_position_adjust_pattern()
                self.setup_timing_pattern()

                precomputed_qr_blanks[self.version] = copy_2d_array(self.modules)

            self.setup_type_info(test, mask_pattern)

            if self.version >= 7:
                self.setup_type_number(test)

            if self.data_cache is None:
                self.data_cache = create_data(
                    self.version, self.error_correction, self.data_list
                )
            self.map_data(self.data_cache, mask_pattern)

        def setup_position_probe_pattern(self, row, col):
            for r in range(-1, 8):

                if row + r <= -1 or self.modules_count <= row + r:
                    continue

                for c in range(-1, 8):

                    if col + c <= -1 or self.modules_count <= col + c:
                        continue

                    if (
                        (0 <= r <= 6 and c in {0, 6})
                        or (0 <= c <= 6 and r in {0, 6})
                        or (2 <= r <= 4 and 2 <= c <= 4)
                    ):
                        self.modules[row + r][col + c] = True
                    else:
                        self.modules[row + r][col + c] = False

        def best_fit(self, start=None):
            """
            Find the minimum size required to fit in the data.
            """
            if start is None:
                start = 1
            check_version(start)

            mode_sizes = mode_sizes_for_version(start)
            buffer = BitBuffer()
            for data in self.data_list:
                buffer.put(data.mode, 4)
                buffer.put(len(data), mode_sizes[data.mode])
                data.write(buffer)

            needed_bits = len(buffer)
            self.version = bisect_left(
                BIT_LIMIT_TABLE[self.error_correction], needed_bits, start
            )
            if self.version == 41:
                raise DataOverflowError()

            # Now check whether we need more bits for the mode sizes, recursing if
            # our guess was too low
            if mode_sizes is not mode_sizes_for_version(self.version):
                self.best_fit(start=self.version)
            return self.version

        def best_mask_pattern(self):
            """
            Find the most efficient mask pattern.
            """
            min_lost_point = 0
            pattern = 0

            for i in range(8):
                self.makeImpl(True, i)

                lost_point1 = lost_point(self.modules)

                if i == 0 or min_lost_point > lost_point1:
                    min_lost_point = lost_point1
                    pattern = i

            return pattern

        def print_tty(self, out=None):
            """
            Output the QR Code only using TTY colors.

            If the data has not been compiled yet, make it first.
            """
            if out is None:
                import sys

                out = sys.stdout

            if not out.isatty():
                raise OSError("Not a tty")

            if self.data_cache is None:
                self.make()

            modcount = self.modules_count
            out.write("\x1b[1;47m" + (" " * (modcount * 2 + 4)) + "\x1b[0m\n")
            for r in range(modcount):
                out.write("\x1b[1;47m  \x1b[40m")
                for c in range(modcount):
                    if self.modules[r][c]:
                        out.write("  ")
                    else:
                        out.write("\x1b[1;47m  \x1b[40m")
                out.write("\x1b[1;47m  \x1b[0m\n")
            out.write("\x1b[1;47m" + (" " * (modcount * 2 + 4)) + "\x1b[0m\n")
            out.flush()

        def print_ascii(self, out=None, tty=False, invert=False):
            """
            Output the QR Code using ASCII characters.

            :param tty: use fixed TTY color codes (forces invert=True)
            :param invert: invert the ASCII characters (solid <-> transparent)
            """
            if out is None:
                out = sys.stdout

            if tty and not out.isatty():
                raise OSError("Not a tty")

            if self.data_cache is None:
                self.make()

            modcount = self.modules_count
            codes = [bytes((code,)).decode("cp437") for code in (255, 223, 220, 219)]
            if tty:
                invert = True
            if invert:
                codes.reverse()

            def get_module(x, y) -> int:
                if invert and self.border and max(x, y) >= modcount + self.border:
                    return 1
                if min(x, y) < 0 or max(x, y) >= modcount:
                    return 0
                return cast(int, self.modules[x][y])

            for r in range(-self.border, modcount + self.border, 2):
                if tty:
                    if not invert or r < modcount + self.border - 1:
                        out.write("\x1b[48;5;232m")  # Background black
                    out.write("\x1b[38;5;255m")  # Foreground white
                for c in range(-self.border, modcount + self.border):
                    pos = get_module(r, c) + (get_module(r + 1, c) << 1)
                    out.write(codes[pos])
                if tty:
                    out.write("\x1b[0m")
                out.write("\n")
            out.flush()

        @overload
        def make_image(self, image_factory: Literal[None] = None, **kwargs) -> GenericImage:
            ...

        @overload
        def make_image(
            self, image_factory: Type[GenericImageLocal] = None, **kwargs
        ) -> GenericImageLocal:
            ...

        def make_image(self, image_factory=None, **kwargs):
            """
            Make an image from the QR Code data.

            If the data has not been compiled yet, make it first.
            """
            if self.data_cache is None:
                self.make()

            if image_factory is not None:
                assert issubclass(image_factory, BaseImage)
            else:
                image_factory = self.image_factory
                if image_factory is None:

                    # Use PIL by default if available, otherwise use PyPNG.
                    image_factory = PilImage if Image else PyPNGImage

            im = image_factory(
                self.border,
                self.modules_count,
                self.box_size,
                qrcode_modules=self.modules,
                **kwargs,
            )

            if im.needs_drawrect:
                for r in range(self.modules_count):
                    for c in range(self.modules_count):
                        if im.needs_context:
                            im.drawrect_context(r, c, qr=self)
                        elif self.modules[r][c]:
                            im.drawrect(r, c)
            if im.needs_processing:
                im.process()

            return im

        # return true if and only if (row, col) is in the module
        def is_constrained(self, row: int, col: int) -> bool:
            return (
                row >= 0
                and row < len(self.modules)
                and col >= 0
                and col < len(self.modules[row])
            )

        def setup_timing_pattern(self):
            for r in range(8, self.modules_count - 8):
                if self.modules[r][6] is not None:
                    continue
                self.modules[r][6] = r % 2 == 0

            for c in range(8, self.modules_count - 8):
                if self.modules[6][c] is not None:
                    continue
                self.modules[6][c] = c % 2 == 0

        def setup_position_adjust_pattern(self):
            pos = pattern_position(self.version)

            for i in range(len(pos)):

                row = pos[i]

                for j in range(len(pos)):

                    col = pos[j]

                    if self.modules[row][col] is not None:
                        continue

                    for r in range(-2, 3):

                        for c in range(-2, 3):

                            if (
                                r == -2
                                or r == 2
                                or c == -2
                                or c == 2
                                or (r == 0 and c == 0)
                            ):
                                self.modules[row + r][col + c] = True
                            else:
                                self.modules[row + r][col + c] = False

        def setup_type_number(self, test):
            bits = BCH_type_number(self.version)

            for i in range(18):
                mod = not test and ((bits >> i) & 1) == 1
                self.modules[i // 3][i % 3 + self.modules_count - 8 - 3] = mod

            for i in range(18):
                mod = not test and ((bits >> i) & 1) == 1
                self.modules[i % 3 + self.modules_count - 8 - 3][i // 3] = mod

        def setup_type_info(self, test, mask_pattern):
            data = (self.error_correction << 3) | mask_pattern
            bits = BCH_type_info(data)

            # vertical
            for i in range(15):

                mod = not test and ((bits >> i) & 1) == 1

                if i < 6:
                    self.modules[i][8] = mod
                elif i < 8:
                    self.modules[i + 1][8] = mod
                else:
                    self.modules[self.modules_count - 15 + i][8] = mod

            # horizontal
            for i in range(15):

                mod = not test and ((bits >> i) & 1) == 1

                if i < 8:
                    self.modules[8][self.modules_count - i - 1] = mod
                elif i < 9:
                    self.modules[8][15 - i - 1 + 1] = mod
                else:
                    self.modules[8][15 - i - 1] = mod

            # fixed module
            self.modules[self.modules_count - 8][8] = not test

        def map_data(self, data, mask_pattern):
            inc = -1
            row = self.modules_count - 1
            bitIndex = 7
            byteIndex = 0

            mask_func1 = mask_func(mask_pattern)

            data_len = len(data)

            for col in range(self.modules_count - 1, 0, -2):

                if col <= 6:
                    col -= 1

                col_range = (col, col - 1)

                while True:

                    for c in col_range:

                        if self.modules[row][c] is None:

                            dark = False

                            if byteIndex < data_len:
                                dark = ((data[byteIndex] >> bitIndex) & 1) == 1

                            if mask_func1(row, c):
                                dark = not dark

                            self.modules[row][c] = dark
                            bitIndex -= 1

                            if bitIndex == -1:
                                byteIndex += 1
                                bitIndex = 7

                    row += inc

                    if row < 0 or self.modules_count <= row:
                        row -= inc
                        inc = -inc
                        break

        def get_matrix(self):
            """
            Return the QR Code as a multidimensional array, including the border.

            To return the array without a border, set ``self.border`` to 0 first.
            """
            if self.data_cache is None:
                self.make()

            if not self.border:
                return self.modules

            width = len(self.modules) + self.border * 2
            code = [[False] * width] * self.border
            x_border = [False] * self.border
            for module in self.modules:
                code.append(x_border + cast(List[bool], module) + x_border)
            code += [[False] * width] * self.border

            return code

        def active_with_neighbors(self, row: int, col: int) -> ActiveWithNeighbors:
            context: List[bool] = []
            for r in range(row - 1, row + 2):
                for c in range(col - 1, col + 2):
                    context.append(self.is_constrained(r, c) and bool(self.modules[r][c]))
            return ActiveWithNeighbors(*context)


def make(data=None, **kwargs):
    qr = QRCode(**kwargs)
    qr.add_data(data)
    return qr.make_image()


class WindowManager (ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class MainWindow (Screen):
    text = StringProperty()
    vdtext = StringProperty()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_click2 (self):
        self.manager.current = 'login'

    def show_load(self):
        Tk().withdraw() # avoids window accompanying tkinter FileChooser
        img = askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        img = cv.imread(img, cv.IMREAD_COLOR)
        barcodes = pyzbar.decode(img)
        txt = barcodes[0].data.decode("utf-8")
        if os.path.exists(txt + '.txt'):
            with open(txt + '.txt', encoding='utf-8') as f:
                self.vdtext = f.readline()
                self.text = f.read()
                self.manager.current = 'mat'

class LogInAdmin (Screen):
    def clc(self):
        print (self.ids)
        password = self.ids['pass'].text
        login = self.ids['log'].text
        print(password, login)
        self.ids['pass'].text = ''
        self.ids['log'].text= ''
        if login == '111' and password == '1111':
            self.manager.current = 'zaeb'


class Zaewindow (Screen):
    def clc(self, **kwargs):
        url = self.ids['url'].text
        hint = self.ids['hint'].text
        print(url, hint)
        self.ids['url'].text = ''
        self.ids['hint'].text= ''
        with open (url[32:38] + '.txt', 'w+') as f:
            f.write(url+'\n'+hint)
        filename = "site.png"
        img = make(url[32:38])
        
        new_file = fd.asksaveasfile(title="Сохранить файл", defaultextension=".png",
                                    filetypes=(("Рисунок", "*.png"),))
        if new_file:
            img.save(new_file.name)
            new_file.close()
        self.manager.current = 'main'
        

class NewMat (Screen):
    label_text = StringProperty()
    vid_url = StringProperty()
    def back (self):
        self.manager.current = 'main'
    def back2 (self):
        Clipboard.copy(self.vid_url)

kv = Builder.load_file('my.kv')

class MyApp(App):
    def build (self):
        return kv


if __name__ == '__main__':
    # data = "goigagoidagoida"
    # # имя конечного файла
    # filename = "site.png"
    # # генерируем qr-код
    # img = make(data)
    # # сохраняем img в файл
    # img.save(filename)
    # print(12)

    MyApp().run()