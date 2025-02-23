using System;
using System.IO;
using System.IO.Ports;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Linq;
using System.Text.RegularExpressions;

class Program{
    static void Main(){
        string portName = "COM12";
        int baudRate = 4000000;

        int cont = 0;
        string directorio = @"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\EEG-tesis\Instrucciones\Registros almacenados\Datos EEG";
        string baseTiempo = "Tiempo";
        string baseFrecuencia = "Frecuencia";
        string filePath_tiempo = GenerarNombreAutoincremental(directorio, baseTiempo); 
        string filePath_frecuencia = GenerarNombreAutoincremental(directorio, baseFrecuencia);
        string start_time = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        string end_time = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"); 

        int port = 5000;
        TcpListener server = new TcpListener(IPAddress.Any, port);
        server.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        server.Start();  
                     
        
        Console.WriteLine("TCP server started on port " + port);
        Thread.Sleep(1000);

        using (SerialPort serialPort = new SerialPort(portName, baudRate)){            
                serialPort.Open();
                Console.WriteLine("Waiting for data...");
                
                bool datosRecibidos = false;                
            
                while (true){
                    try{
                        using (TcpClient client = server.AcceptTcpClient()){
                            using (NetworkStream stream = client.GetStream()){
                                Console.WriteLine("Client connected succesfully!");   

                                EscribirLinea(filePath_tiempo, "StartTime; EndTime; Channel1; Channel2; Channel3; Channel4; Channel5");
                                EscribirLinea(filePath_frecuencia, "StartTime; EndTime; Channel1; Channel2; Channel3; Channel4; Channel5");

                                while (true){                                    
                                    string line = serialPort.ReadLine();
                                    string[] valoresString = line.Split(',');                       

                                    if (valoresString.Length == 645){ 
                                        Console.WriteLine("FFT");
                                        Console.WriteLine(cont);
                                        cont = 0;
                                        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");                                                                           
                                        bool allgood = true;

                                        for (int i = 0; i < valoresString.Length; i++){
                                            if (!double.TryParse(valoresString[i], out double value)){   
                                                allgood = false;                                                                                         
                                            }
                                        }

                                        if (allgood){                                            
                                            string datos = $"{timestamp};{string.Join(";", valoresString)}\n";
                                            string[] num_datos = datos.Split(';');
                                            byte[] data = Encoding.ASCII.GetBytes(datos);
                                            stream.Write(data, 0, data.Length);                                          
                                            // writer.WriteLine($"{timestamp}; {valoresLeidos[0]}; {valoresLeidos[1]}; {valoresLeidos[2]}; {valoresLeidos[3]}; {valoresLeidos[4]}");
                                            // writer.Flush(); 

                                            if (!datosRecibidos){
                                                datosRecibidos = true;
                                                Console.WriteLine("Data is being collected... Starting plotting and storage process\n");
                                            }
                                        }
                                    }else if(valoresString.Length == 6 && string.Equals(valoresString[^1].Trim(),"OLD")){
                                        cont++;
                                        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");                              
                                        bool allgood = true;

                                        for (int i = 0; i < valoresString.Length - 1; i++){
                                            if (!double.TryParse(valoresString[i], out double value)){
                                                allgood = false;                                            
                                            }
                                        }

                                        if (allgood){                                                                                        
                                            string datos = $"{timestamp};{string.Join(";", valoresString)}\n";
                                            string[] num_datos = datos.Split(';');
                                            byte[] data = Encoding.ASCII.GetBytes(datos);                                            
                                            stream.Write(data, 0, data.Length);                                          
                                            // writer.WriteLine($"{timestamp}; {valoresLeidos[0]}; {valoresLeidos[1]}; {valoresLeidos[2]}; {valoresLeidos[3]}; {valoresLeidos[4]}");
                                            // writer.Flush();                                        
                                        }
                                    }
                                }
                            }
                        }
                    }
                    catch (TimeoutException)
                    {
                        Console.WriteLine("Timeout error on serial port.");
                    }
                    catch (SocketException e)
                    {
                        Console.WriteLine($"SocketException: {e.Message}");
                    }
                }
            // }
        }
    }

    static string GenerarNombreAutoincremental(string directorio, string baseNombre){
        if (!Directory.Exists(directorio)){
            Console.WriteLine("No directory");
            Directory.CreateDirectory(directorio);
        }
        
        var archivos = Directory.GetFiles(directorio, baseNombre + "_*.csv")
            .Select(Path.GetFileNameWithoutExtension)
            .Where(f => Regex.IsMatch(f, $"^{baseNombre}_\\d+$"))
            .Select(f => int.Parse(f.Substring(baseNombre.Length + 1)))
            .ToList();
        
        int nuevoNumero = archivos.Any() ? archivos.Max() + 1 : 1;
        return Path.Combine(directorio, $"{baseNombre}_{nuevoNumero}.csv");
    }

    static void EscribirLinea(string filePath, string info){
        using (StreamWriter writer = new StreamWriter(filePath)){
            writer.WriteLine(info);
        }
    } 
}