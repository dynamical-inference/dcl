import abc
import math
from dataclasses import dataclass

from jaxtyping import jaxtyped
from typeguard import typechecked

from dcl.utils.configurable import Configurable
from dcl.utils.datatypes import config_field


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class TemperatureScheduler(Configurable, abc.ABC):

    max_epochs: int = config_field(default=100)  # Maximum number of epochs to
    initial_temp: float = config_field(
        default=1.0)  # Starting temperature value
    min_temp: float = config_field(
        default=1e-16)  # Minimum temperature value that can be reached

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be greater than 0.")

        # temperatures must be > 0
        if self.initial_temp <= 0:
            raise ValueError("initial_temp must be greater than 0.")
        if self.min_temp <= 0:
            raise ValueError("min_temp must be greater than 0.")
        if self.initial_temp < self.min_temp:
            raise ValueError("initial_temp must be greater than min_temp.")

    @abc.abstractmethod
    def get_temperature(self, epoch: int) -> float:
        pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class DelayedTemperatureScheduler(TemperatureScheduler):
    delay_epoch: int = config_field(
        default=0
    )  # Number of epochs to wait before starting temperature scheduling

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if self.delay_epoch < 0:
            raise ValueError("delay_epoch must be greater or equal to 0.")
        if self.delay_epoch > self.max_epochs:
            raise ValueError("delay_epoch must be less than max_epochs.")

    @jaxtyped(typechecker=typechecked)
    def get_temperature(self, epoch: int) -> float:
        if epoch < self.delay_epoch:
            temperature = self.initial_temp
        else:
            temperature = self._get_temperature(epoch)
        # bound with min and initial temperature
        temperature = max(self.min_temp, temperature)
        temperature = min(self.initial_temp, temperature)
        return temperature

    @abc.abstractmethod
    def _get_temperature(self, epoch: int) -> float:
        pass


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class LinearTemperatureScheduler(DelayedTemperatureScheduler):

    def _get_temperature(self, epoch: int) -> float:
        # linear scale means f(x) = ax + b, where x is the epoch, f(x) is the temperature
        # b is the initial temperature, a is the slope

        # slope = (self.initial_temp - self.min_temp)
        slope = (self.initial_temp - self.min_temp)
        # correct slope for reaching min_temp in (max_epochs-delay_epochs) steps
        corrected_slope = slope / (self.max_epochs - self.delay_epoch)

        # correct epoch for delay (correcting x)
        corrected_epoch = epoch - self.delay_epoch

        # calculate temperature
        epoch_temp = self.initial_temp - corrected_slope * corrected_epoch

        return epoch_temp


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ExponentialTemperatureScheduler(DelayedTemperatureScheduler):

    def _get_temperature(self, epoch: int) -> float:
        return self.initial_temp * math.exp(
            -0.1 * (epoch - self.delay_epoch) /
            (self.max_epochs - self.delay_epoch))


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class CyclingTemperatureScheduler(DelayedTemperatureScheduler):
    num_cycles: int = config_field(
        default=1)  # Number of temperature cycles to complete over training

    def _get_temperature(self, epoch: int) -> float:
        cycle_length = (self.max_epochs - self.delay_epoch) // self.num_cycles
        cycle_position = (epoch - self.delay_epoch) % cycle_length
        cycle_progress = cycle_position / cycle_length
        return self.min_temp + (self.initial_temp - self.min_temp) * (
            0.5 * (1 + math.cos(2 * math.pi * cycle_progress - math.pi / 2)))


@jaxtyped(typechecker=typechecked)
@dataclass(kw_only=True)
class ConstantTemperatureScheduler(TemperatureScheduler):

    def get_temperature(self, epoch: int) -> float:
        return self.initial_temp
