using System;
using System.IO.Ports;

class Program{
    static void Main(){
        string portName = "COM12"; // Cambia esto al nombre de tu puerto
        int baudRate = 115200; // Ajusta la velocidad en baudios

        // Especifica la ruta donde quieres guardar el archivo CSV
        string filePath = @"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\Instrucciones\Registros almacenados\Datos EEG\Pruebas\Datos_1.csv"; 
        
        using (SerialPort serialPort = new SerialPort(portName, baudRate)){            
            using (StreamWriter writer = new StreamWriter(filePath)){            
                writer.WriteLine("Time; Channel1; Channel2; Channel3; Channel4; Channel5");                
                serialPort.Open();

                Console.WriteLine("Waiting for data...");

                bool datosRecibidos = false; // Bandera para controlar la correcta llegada de señales

                while (true){
                    try{
                        string line = serialPort.ReadLine(); // Lee la línea entrante
                        string[] valoresString = line.Split(','); // Divide la línea en valores

                        if (valoresString.Length != 5){                            
                            continue; // Salta al siguiente ciclo si no se reciben 5 valores
                        }

                        // Obtener el tiempo actual del sistema
                        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");

                        // Crear un arreglo para almacenar los valores leídos
                        double[] valoresLeidos = new double[valoresString.Length];

                        // Convertir los valores de string a double
                        bool allgood = true;
                        for (int i = 0; i < valoresString.Length; i++){
                            if (double.TryParse(valoresString[i], out double value)){
                                valoresLeidos[i] = value*1e-6; // Almacena el valor en el arreglo
                            }
                            else{                                
                                allgood = false;
                            }
                        }

                        if(allgood){
                            // Mostrar los valores almacenados  
                            //Console.WriteLine("Valores");                      
                            for(int i = 0; i < valoresLeidos.Length ;i++){
                                //Console.WriteLine(valoresLeidos[i]); 
                                valoresLeidos[i] = Math.Round(valoresLeidos[i],6);
                            }                     
                            // Escribir los datos en el archivo CSV
                            writer.WriteLine($"{timestamp}; {valoresLeidos[0]}; {valoresLeidos[1]}; {valoresLeidos[2]}; {valoresLeidos[3]}; {valoresLeidos[4]}");
                            // Forzar el guardado de datos en el archivo
                            writer.Flush();

                            if (!datosRecibidos){
                                datosRecibidos = true; // Actualiza la bandera
                                Console.WriteLine("Data is beeing collected well...starting storage process"); // Señal a Python
                            }
                        }
                    }
                    catch (TimeoutException){
                        Console.WriteLine("Reading timeout error");
                    }
                }            
            }
        }
    }
}
