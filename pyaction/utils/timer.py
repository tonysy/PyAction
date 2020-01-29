#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# -*- coding: utf-8 -*-

from time import perf_counter
from typing import Optional


class Timer:
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """

    def __init__(self):
        self.reset()
        # Start time
        self._begin = perf_counter()
        self._past_durations = []

    def reset(self):
        """
        Reset the timer.
        """
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0

    def pause(self):
        """
        Pause the timer.
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        """
        Returns:
            bool: whether the timer is currently paused
        """
        return self._paused is not None

    def resume(self):
        """
        Resume the timer.
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None

    def seconds(self) -> float:
        """
        Returns:
            (float): the total number of seconds since the start/reset of the
                timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused  # type: ignore
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def past(self) -> float:
        """
        For PyAction

        Returns:
            (float): the total number of seconds since the start
        """
        return perf_counter() - self._begin

    def data_toc(self) -> float:
        """
        For PyAction

        Returns:
            (float): data processing times
        """
        self.data_time = perf_counter() - self._start

    def forward_toc(self) -> float:
        """
        For PyAction

        Returns:
            (float): network forward times
        """
        self.forward_time = perf_counter() - (self._start + self.data_time)

    def loss_toc(self) -> float:
        """
        For PyAction

        Returns:
            (float): loss computation times
        """
        self.loss_time = perf_counter() - (
            self._start + self.data_time + self.forward_time
        )

    def backward_toc(self) -> float:
        """
        For PyAction

        Returns:
            (float): network backward times
        """
        self.backward_time = perf_counter() - (
            self._start + self.data_time + self.forward_time + self.loss_time
        )
