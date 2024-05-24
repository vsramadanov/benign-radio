import pyaudio
import time
import numpy as np
from dataclasses import dataclass

from simulation.unit import SimUnit

from dsp.resampling import PolyResampler
from dsp.resampling import PolyResamplerConfig

from scipy.signal import hilbert


class CallbackPlayer(SimUnit):
    def __init__(self, src: np.array) -> None:
        super().__init__()

        self.data = src.astype(np.int16).tobytes()
        self.len = len(src)
        self.frame = 0
        self.logger.info(f'initiated with {self.len} frames')

    def __call__(self, in_data, frame_count, time_info, status):

        begin = self.frame
        end = min(self.frame + frame_count, self.len)

        self.frame = end

        data = self.data[2*begin:2*end]  # int16 contains 2 bytes
        if (end < self.len):
            return data, pyaudio.paContinue

        return data, pyaudio.paComplete


class CallbackRecorder(SimUnit):
    def __init__(self, dst: list) -> None:
        super().__init__()

        self.data = dst
        self.logger.info('Recorder created')

    def __call__(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16)
        self.data.append(data)
        return None, pyaudio.paContinue


@dataclass
class AudioChannelConfig:
    fs: int
    channels: int
    format: str
    chunk: int


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

        self.upsampler = PolyResampler(
            config=PolyResamplerConfig(
                fin=self.params.fs,
                fout=self.config.fs,
            )
        )
        self.downsampler = PolyResampler(
            config=PolyResamplerConfig(
                fin=self.config.fs,
                fout=self.params.fs,
            )
        )

    def __enter__(self):
        self.ctx = pyaudio.PyAudio()
        self.logger.info('PyAudio context created')

        return self

    def __exit__(self, type, value, traceback):
        self.ctx.terminate()
        self.logger.info('PyAudio context terminated')

        if not (type is None and value is None and traceback is None):
            self.logger.error(
                f'type: {type}, value: {value}, traceback: {traceback}')

    def process_raw(self, tx_signal) -> np.array:
        config = self.config
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
        self.logger.info('receiver started')
        stream_play.start_stream()
        self.logger.info('transmiter started')

        while stream_play.is_active() and stream_record.is_active():
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

    def process(self, tx_signal) -> np.array:
        upsampled = self.upsampler.process(tx_signal)
        upsampled_scaled = (2**14 * upsampled).astype(np.int16)

        raw_recv = self.process_raw(upsampled_scaled)
        recv = hilbert(raw_recv.astype(np.float64))

        recv_downsampled = self.downsampler.process(recv)

        return recv_downsampled
