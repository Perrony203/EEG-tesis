#ifndef SVMARMSLEGS_H
#define SVMARMSLEGS_H
 /**
 * @brief Realiza la predicción de clasificación de intención de movimiento de miembros superiores e inferiores.
 *
 * Esta función predice el resultado de la clasificación de intención de movimiento
 * de miembros superiores e inferiores basado en el vector de entrada que corresponde
 * a las características para los cinco canales de medición.
 *
 * @param x puntero al vector de entrada.
 * @return El resultado de la predicción, siendo 0 una intención de movimiento de piernas y 1 una intención de movimiento de brazos.
 */
int predict_AL(double *x);

#endif // SVMARMSLEGS_H
