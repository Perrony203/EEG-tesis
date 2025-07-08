#ifndef SVMMOVEMENT_H
#define SVMMOVEMENT_H
 /**
 * @brief Realiza la predicción de detección de intención de movimiento en algún miembro.
 *
 * Esta función predice el resultado de la clasificación de intención de movimiento o
 * el continuo estado basal (inactividad) basado en el vector de entrada que corresponde
 * a las características para los cinco canales de medición.
 *
 * @param x puntero al vector de entrada.
 * @return El resultado de la predicción, siendo 0 una intención de movimiento de piernas y 1 una intención de movimiento de brazos.
 */
int predict_M(double *x);

#endif // SVMMOVEMENT_H
