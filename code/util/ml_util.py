
"""
General utility functions for machine learning. Borrowed and extended from https://github.com/vitchyr/rlkit
"""
import abc
import math
from collections import deque
import numpy as np

class CategoricalSchedule:
    def __init__(self, left_boundaries, values):
        assert(len(left_boundaries) == len(values))
        assert(all(_ > 0 for _ in left_boundaries))
        assert(np.all(np.diff(left_boundaries) > 0))
        self.boundaries_and_values_reversed = list(zip(left_boundaries, values))
        
    def get_value(self, t):
        for boundary, value in self.boundaries_and_values_reversed:
            if t > boundary: return value
            continue
        raise ValueError("Input time is lower than smallest boundary!")

    def update(self, val):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.boundaries_and_values_reversed})"
        
class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass

    def update(self, val):
        pass    

class ConstantSchedule(ScalarSchedule):
    def __init__(self, value):
        self._value = value

    def get_value(self, t):
        return self._value

    def __repr__(self):
        return f"{self.__class__.__name__}({self._value})"

class LinearSchedule(ScalarSchedule):
    """
    Linearly interpolate and then stop at a final value.
    """
    def __init__(
            self,
            init_value=1.,
            final_value=1.,
            ramp_duration=1000,
    ):
        self._init_value = init_value
        self._final_value = final_value
        self._ramp_duration = ramp_duration

    def get_value(self, t):
        return (
            self._init_value
            + (self._final_value - self._init_value)
            * min(1.0, t * 1.0 / self._ramp_duration)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(init_value={self._init_value}, final_value={self._final_value}, ramp_duration={self._ramp_duration})"
    
    
class IntLinearSchedule(LinearSchedule):
    """
    Same as RampUpSchedule but round output to an int
    """
    def get_value(self, t):
        return int(super().get_value(t))


class PiecewiseLinearSchedule(ScalarSchedule):
    """
    Given a list of (x, t) value-time pairs, return value x at time t,
    and linearly interpolate between the two
    """
    def __init__(
            self,
            x_values,
            y_values,
    ):
        self._x_values = x_values
        self._y_values = y_values

    def get_value(self, t):
        return np.interp(t, self._x_values, self._y_values)


class IntPiecewiseLinearSchedule(PiecewiseLinearSchedule):
    def get_value(self, t):
        return int(super().get_value(t))


def none_to_infty(bounds):
    if bounds is None:
        bounds = -math.inf, math.inf
    lb, ub = bounds
    if lb is None:
        lb = -math.inf
    if ub is None:
        ub = math.inf
    return lb, ub


class StatConditionalSchedule(ScalarSchedule):
    """
    Every time a (running average of the) statistic dips is above a threshold,
    add `delta` to the outputted value.

    If the statistic is below a threshold, subtract `delta` to the
    outputted value.
    """
    def __init__(
            self,
            init_value=None,
            stat_bounds=None,
            running_average_length=None,
            delta=1,
            value_bounds=None,
            statistic_name=None,
            min_num_stats=None,
            min_time_gap_between_value_changes=0,
    ):
        """
        :param init_value: Initial value outputted
        :param stat_bounds: (min, max) values for the stat. When the running
        average of the stat exceeds this threshold, the outputted value changes.
        :param running_average_length: How many stat values to average. Not
        updates occur until this many samples are taken.
        :param statistic_name: Name of the statistic to follow.
        :param delta: How much to add to the output value when the statistic
        is above the threshold.
        :param value_bounds: (min, max) ints for the value. If None, then the
        outputted value can grow and shrink arbitrarily.
        :param min_time_gap_between_value_changes: At least this much time
        must pass before updating the outputted value again.
        """
        if min_num_stats is None:
            min_num_stats = running_average_length
        if value_bounds is None:
            value_bounds = -math.inf, math.inf
        value_lb, value_ub = none_to_infty(value_bounds)
        stat_lb, stat_ub = none_to_infty(stat_bounds)

        assert min_num_stats <= running_average_length
        assert stat_lb < stat_ub
        assert value_lb < value_ub
        assert isinstance(min_time_gap_between_value_changes, int)

        self._value = init_value
        self.stat_lb, self.stat_ub = stat_lb, stat_ub
        self._stats = deque(maxlen=running_average_length)
        self.delta = delta
        self.value_lb, self.value_ub = value_lb, value_ub
        self.statistic_name = statistic_name
        self.min_number_stats = min_num_stats
        self.min_gap_between_updates = min_time_gap_between_value_changes
        self._t = -1
        self._last_update_t = None

    def get_value(self, t):
        self._t = t
        return self._value

    def update(self, statistics):
        stat = statistics[self.statistic_name]
        self._stats.append(stat)
        if len(self._stats) < self.min_number_stats:
            return

        if self._last_update_t is not None:
            if self._t - self._last_update_t < self.min_gap_between_updates:
                return

        mean = np.mean(self._stats)
        if mean > self.stat_ub:
            self._value += self.delta
            self._last_update_t = self._t
        elif mean < self.stat_lb:
            self._value -= self.delta
            self._last_update_t = self._t
        else:
            pass
        
        if self.value_ub is not None:
            self._value = min(self.value_ub, max(self.value_lb, self._value))

    def __repr__(self):
        return f"{self.__class__.__name__}(t={self._t}, value={self._value}, stat_bounds={(self.stat_lb, self.stat_ub)}, delta={self.delta}, value_bounds={(self.value_lb, self.value_ub)})"

class ConditionalBooleanSchedule:
    def __init__(self, false_value=None, true_value=None):
        self._t = 0
        self._true_value = true_value
        self._false_value = false_value
        self._value = self._true_value
        
    def get_value(self, t):
        self._t = t
        return self._value

    def update(self, boolean):
        assert(boolean in (True, False))
        if boolean: self._value = self._true_value
        else: self._value = self._false_value
        self._t += 1

    def __repr__(self):
        return f"{self.__class__.__name__}(true_value={self._true_value}, false_value={self._false_value})"


class LessThanThresholdSchedule(ConditionalBooleanSchedule):
    def __init__(self, below_value, above_value, threshold, statistic_name):
        super().__init__(false_value=above_value, true_value=below_value)
        self._value = above_value
        self._below_value = below_value
        self._above_value = above_value
        self._threshold = threshold
        self._statistic_name = statistic_name

    def update(self, statistics):
        stat = statistics[self._statistic_name]
        super().update(stat < self._threshold)
