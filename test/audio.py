import pyaudio
import time
import numpy as np
from dataclasses import dataclass
from fractions import Fraction

from simulation.unit import SimUnit

from scipy.signal import hilbert


class CallbackPlayer(SimUnit):
    def __init__(self, src: np.array) -> None:
        super().__init__()

        self.raw_data = src.astype(np.int16).tobytes()
        self.len = len(src)
        self.idx = 0
        self.logger.info(f'initiated with {self.len} samples')

    def __call__(self, in_data, frame_count, time_info, status):

        start = self.idx
        stop = min(self.idx + frame_count, self.len)

        self.idx = stop

        data = self.raw_data[2*start:2*stop]  # int16 contains 2 bytes

        if (stop < self.len):
            self.logger.debug(f'requested {frame_count} samples, '
                              f'provided: {stop - start}\n'
                              f'\ttime info: {time_info}, status: {status}')
            return data, pyaudio.paContinue

        self.logger.debug(f'requested {frame_count} samples, '
                          f'provided: {stop - start}. stream is finnished\n'
                          f'\ttime info: {time_info}, status: {status}')
        return data, pyaudio.paComplete


class CallbackRecorder(SimUnit):
    def __init__(self, dst: list) -> None:
        super().__init__()

        self.dst = dst
        self.logger.info('Recorder created')

    def __call__(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        self.logger.debug(f'received {frame_count} samples, {len(in_data)} bytes\n'
                          f'\ttime info: {time_info}, status: {status}')
        self.dst.append(data)
        return None, pyaudio.paContinue


@dataclass
class AudioChannelConfig:
    fs: int
    channels: int
    format: str
    chunk: int
    tx_zero_prefix: float


class AudioChannel(SimUnit):
    def __init__(self, config: AudioChannelConfig) -> None:
        super().__init__()

        if config.fs <= 2 * (self.params.fc + self.params.fs / 2):
            self.logger.critical(f'Unable to reproduce signal '
                                 f'with carrier frequency {self.params.fc} '
                                 f'and baseband {self.params.fs}'
                                 f'on sampling frequency {config.fs}')
            assert False

        self.config = config
        self.config.format = getattr(pyaudio, 'pa' + self.config.format)
        self.logger.info(f'created with config: {config}')

        ratio = Fraction(
            numerator=self.config.fs,
            denominator=self.params.fs,
        )

        assert ratio.denominator == 1

        self.upscale_factor = ratio.numerator
        self.downscale_factor = ratio.numerator

    def __enter__(self):
        self.ctx = pyaudio.PyAudio()
        self.logger.info('PyAudio context created')

        return self

    def __exit__(self, type, value, tb):
        self.ctx.terminate()
        self.logger.info('PyAudio context terminated')

        if not (type is None and value is None and tb is None):
            self.logger.error(f'type: {type}, value: {value}, traceback: {tb}')

    def __add_zero_prefix(self, tx_signal):
        prefix_len = int(self.config.tx_zero_prefix * self.config.fs)

        return np.concatenate((
            np.zeros(prefix_len, dtype=np.int16),
            tx_signal
        ), axis=0)

    def process_raw(self, tx_signal) -> np.array:
        config = self.config

        tx_signal = self.__add_zero_prefix(tx_signal)
        stream_play = self.ctx.open(format=config.format,
                                    channels=config.channels,
                                    rate=config.fs,
                                    output=True,
                                    stream_callback=CallbackPlayer(src=tx_signal))

        records = []
        stream_record = self.ctx.open(format=config.format,
                                      channels=config.channels,
                                      rate=config.fs,
                                      input=True,
                                      stream_callback=CallbackRecorder(dst=records))

        stream_record.start_stream()
        self.logger.info(f'receiver started')
        stream_play.start_stream()
        self.logger.info('transmiter started')

        while stream_play.is_active():
            if not stream_record.is_active():
                self.logger.critical('failure in audio receiver')
            time.sleep(.1)

        self.logger.info('transmittion has ended')

        stream_play.stop_stream()
        stream_play.close()
        self.logger.info('transmiter stopped')

        stream_record.stop_stream()
        stream_record.close()
        self.logger.info('receiver stopped')

        raw_recv = np.concatenate(records, axis=0)
        self.logger.debug(f'received {raw_recv.shape[0]} samples, '
                          f'{raw_recv.shape[0] / self.config.fs} seconds of data')
        return raw_recv
