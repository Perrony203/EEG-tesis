#include "standarScalerLR.h"
#include <math.h>

void scaler_init_LR(StandardScalerLR* scaler) {
    scaler->current_size = 0;
    scaler->index = 0;
    for (int i = 0; i < MAX_SAMPLES_LR; i++) {
        for (int j = 0; j < N_FEATURES_LR; j++) {
            scaler->buffer[i][j] = 0.0;
        }
    }
}

void scaler_add_sample_LR(StandardScalerLR* scaler, const double sample[N_FEATURES_LR]) {
    for (int j = 0; j < N_FEATURES_LR; j++) {
        scaler->buffer[scaler->index][j] = sample[j];
    }
    scaler->index = (scaler->index + 1) % MAX_SAMPLES_LR;
    if (scaler->current_size < MAX_SAMPLES_LR) {
        scaler->current_size++;
    }
}

void scaler_compute_LR(const StandardScalerLR* scaler, double mean[N_FEATURES_LR], double std[N_FEATURES_LR]) {
    for (int j = 0; j < N_FEATURES_LR; j++) {
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

void scaler_transform_LR(StandardScalerLR* scaler, const double sample[N_FEATURES_LR], double mean[N_FEATURES_LR], double std[N_FEATURES_LR]) {
    for (int j = 0; j < N_FEATURES_LR; j++) {
        if (std[j] != 0.0)
        	scaler->output[j] = ((sample[j] - mean[j]) / std[j]);
        else
        	scaler->output[j] = 0.0;
    }
}
