#ifndef STANDARSCALERLR_H
#define STANDARSCALERLR_H
#include <stddef.h>

#define MAX_SAMPLES_LR 1
#define N_FEATURES_LR 20   // número de características por muestra

typedef struct {
	double buffer[MAX_SAMPLES_LR][N_FEATURES_LR];
	int current_size;
	int index;
    double output[N_FEATURES_LR];
} StandardScalerLR;

void scaler_init_LR(StandardScalerLR* scaler);
void scaler_add_sample_LR(StandardScalerLR* scaler, const double sample[N_FEATURES_LR]);
void scaler_compute_LR(const StandardScalerLR* scaler, double mean[N_FEATURES_LR], double std[N_FEATURES_LR]);
void scaler_transform_LR(StandardScalerLR* scaler, const double sample[N_FEATURES_LR], double mean[N_FEATURES_LR], double std[N_FEATURES_LR]);

#endif /* SCALER_STANDARSCALERLR_H_ */
