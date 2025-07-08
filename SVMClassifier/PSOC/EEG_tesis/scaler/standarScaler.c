#include "standarScaler.h"
#include <math.h>

void scaler_init(StandardScaler* scaler) {
    scaler->current_size = 0;
    scaler->index = 0;
    for (int i = 0; i < MAX_SAMPLES; i++) {
        for (int j = 0; j < N_FEATURES; j++) {
            scaler->buffer[i][j] = 0.0;
        }
    }
}

void scaler_add_sample(StandardScaler* scaler, const double sample[N_FEATURES]) {
    for (int j = 0; j < N_FEATURES; j++) {
        scaler->buffer[scaler->index][j] = sample[j];
    }
    scaler->index = (scaler->index + 1) % MAX_SAMPLES;
    if (scaler->current_size < MAX_SAMPLES) {
        scaler->current_size++;
    }
}

void scaler_compute(const StandardScaler* scaler, double mean[N_FEATURES], double std[N_FEATURES]) {
    for (int j = 0; j < N_FEATURES; j++) {
        mean[j] = 0.0;
        std[j] = 0.0;

        for (int i = 0; i < scaler->current_size; i++) {
            mean[j] += scaler->buffer[i][j];
        }
        mean[j] /= scaler->current_size;

        for (int i = 0; i < scaler->current_size; i++) {
            std[j] += (scaler->buffer[i][j] - mean[j])*(scaler->buffer[i][j] - mean[j]);
        }
        std[j] = sqrt(std[j] / scaler->current_size);
    }
}

void scaler_transform(StandardScaler* scaler, const double sample[N_FEATURES], double mean[N_FEATURES], double std[N_FEATURES]) {
    for (int j = 0; j < N_FEATURES; j++) {
        if (std[j] != 0.0)
        	scaler->output[j] = ((sample[j] - mean[j]) / std[j]);
        else
        	scaler->output[j] = 0.0;
    }
}
