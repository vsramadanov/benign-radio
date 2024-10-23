
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <alsa/asoundlib.h>

#define BUFFER_SIZE 2048

void captureAudio(const char *device, unsigned int rate) {
    int err;
    snd_pcm_t *pcm_handle;
    snd_pcm_hw_params_t *params;
    char *buffer;

    if ((err = snd_pcm_open(&pcm_handle, device, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "unable to open pcm device for capture: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }

    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm_handle, params);
    snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_rate_near(pcm_handle, params, &rate, 0);
    snd_pcm_hw_params_set_channels(pcm_handle, params, 1);

    if ((err = snd_pcm_hw_params(pcm_handle, params)) < 0) {
        fprintf(stderr, "unable to set hw parameters: %s\n", snd_strerror(err));
        exit(EXIT_FAILURE);
    }

    buffer = malloc(BUFFER_SIZE);
    if (!buffer) {
        fprintf(stderr, "unable to allocate buffer\n");
        exit(EXIT_FAILURE);
    }

    while (1) {
        int frame_count = BUFFER_SIZE / 2;
        snd_pcm_sframes_t read_frames = snd_pcm_readi(pcm_handle, buffer, frame_count);
        if (read_frames < 0) {
            fprintf(stderr, "snd_pcm_readi failed: %s\n", snd_strerror(read_frames));
            break;
        }
        
        int send_bytes = write(STDOUT_FILENO, buffer, read_frames * sizeof(int16_t));
        fprintf(stderr, "captured and send %ld frames (%d bytes)\n", read_frames, send_bytes);
    }

    free(buffer);
    snd_pcm_drain(pcm_handle);
    snd_pcm_close(pcm_handle);
}

int main(int argc, char *argv[]) {
    const char *device = "default";
    unsigned int rate = 44100;

    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'},
        {"rate", required_argument, 0, 'r'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:r:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'd':
                device = optarg;
                break;
            case 'r':
                rate = (unsigned int)atoi(optarg);
                break;
            default:
                fprintf(stderr, "Usage: %s [--device <device>] [--rate <rate>]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    captureAudio(device, rate);
    return EXIT_SUCCESS;
}
