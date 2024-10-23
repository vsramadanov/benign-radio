import numpy as np
import subprocess

from dataclasses import dataclass
from simulation.unit import SimUnit


@dataclass
class AudioChannelConfig:
    rate: int
    channels: int
    format: str
    device: str


class RawAudioChannel(SimUnit):
    def __init__(self, config: AudioChannelConfig) -> None:
        super().__init__()

        if config.rate <= 2 * (self.params.fc + self.params.fs / 2):
            self.logger.critical(f'Unable to reproduce signal '
                                 f'with carrier frequency {self.params.fc} '
                                 f'and baseband {self.params.fs}'
                                 f'on sampling frequency {config.rate}')
            assert False

        if config.channels not in (1, 2):
            self.logger.critical(
                f'{config.channels} channels seems to be not supported by audio HW')
            assert False

        self.config = config
        self.logger.info(f'created with config: {config}')

    def __enter__(self):

        self.audio_in_process = self.__run_audio_in()
        self.audio_out_process = self.__run_audio_out()

        return self

    def __exit__(self, type, value, tb):

        self.audio_in_process.stdout.close()
        self.audio_in_process.stderr.close()
        self.audio_out_process.stderr.close()

        self.audio_in_process.terminate()
        self.audio_out_process.terminate()

        if not (type is None and value is None and tb is None):
            self.logger.error(f'type: {type}, value: {value}, traceback: {tb}')

    def __run_audio_in(self):
        return subprocess.Popen(
            ['./csound/audio_in',
                f'--rate={self.config.rate}', f'--device={self.config.device}'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def __run_audio_out(self):
        return subprocess.Popen(
            ['./csound/audio_out',
                f'--rate={self.config.rate}', f'--device={self.config.device}'],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def route_audio(self, input: np.array) -> np.array:
        if self.config.channels == 1:
            input = np.repeat(input, 2)
        recorded_data = []

        b = self.audio_out_process.stdin.write(input.tobytes())
        self.logger.debug(f"write {b} bytes to audio_out, close the pipe")
        self.audio_out_process.stdin.close()

        while self.audio_out_process.poll() is None:
            output = self.audio_in_process.stdout.read(4096)
            self.logger.debug(f"read {len(output)} bytes from audio_in")
            if not output:
                break

            recorded_data.append(np.frombuffer(output, dtype=np.int16))

        output = self.audio_in_process.stdout.read(4096)
        self.logger.debug(f"read {len(output)} bytes from audio_in")
        recorded_data.append(np.frombuffer(output, dtype=np.int16))

        return np.concatenate(recorded_data, axis=0)
