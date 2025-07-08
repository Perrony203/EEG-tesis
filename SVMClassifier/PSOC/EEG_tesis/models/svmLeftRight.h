#ifndef SVMLEFTRIGHT_H
#define SVMLEFTRIGHT_H
 /**
 * @brief Realiza la predicción de clasificación de intención de movimiento de miembros izquierdos y derechos.
 *
 * Esta función predice el resultado de la clasificación de intención de movimiento
 * de miembros izquierdos y derechos basado en el vector de entrada que corresponde
 * a las características para los cinco canales de medición.
 *
 * @param x puntero al vector de entrada.
 * @return El resultado de la predicción, siendo 0 una intención de movimiento de miembros izquierdos y 1 una intención de movimiento de miembros derechos.
 */
int predict_LR(double *x);

#endif // SVMLEFTRIGHT_H
