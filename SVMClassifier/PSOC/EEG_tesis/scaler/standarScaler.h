#ifndef STANDARSCALER_H
#define STANDARSCALER_H
#include <stddef.h>

#define MAX_SAMPLES 1  // número máximo de muestras a usar para escalamiento
#define N_FEATURES 40   // número de características por muestra

typedef struct {
	double buffer[MAX_SAMPLES][N_FEATURES];
	int current_size;
	int index;
    double output[N_FEATURES];
} StandardScaler;

void scaler_init(StandardScaler* scaler);
void scaler_add_sample(StandardScaler* scaler, const double sample[N_FEATURES]);
void scaler_compute(const StandardScaler* scaler, double mean[N_FEATURES], double std[N_FEATURES]);
void scaler_transform(StandardScaler* scaler, const double sample[N_FEATURES], double mean[N_FEATURES], double std[N_FEATURES]);

#endif /* SCALER_STANDARSCALER_H_ */
