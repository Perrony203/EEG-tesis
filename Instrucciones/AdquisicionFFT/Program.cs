using System;
using System.IO.Ports;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

class Program{
    static void Main(){
        string portName = "COM12";
        int baudRate = 4000000;
        
        string filePath = @"D:\Universidad\Trabajo de grado\Desarrollo prototipo\Código\Instrucciones\Registros almacenados\Datos EEG\Datos_1.csv"; 

        int port = 5000;
        TcpListener server = new TcpListener(IPAddress.Any, port);
        server.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);

        server.Start();               
        
        Console.WriteLine("TCP server started on port " + port);
        Thread.Sleep(1000);

        using (SerialPort serialPort = new SerialPort(portName, baudRate)){
            // using (StreamWriter writer = new StreamWriter(filePath)){
                // writer.WriteLine("Time; Channel1; Channel2; Channel3; Channel4; Channel5");
                serialPort.Open();
                Console.WriteLine("Waiting for data...");
                
                bool datosRecibidos = false;                

                while (true){
                    try{
                        using (TcpClient client = server.AcceptTcpClient()){
                            using (NetworkStream stream = client.GetStream()){
                                Console.WriteLine("Client connected succesfully!");                                                                 
                                while (true){                                    
                                    string line = serialPort.ReadLine();
                                    string[] valoresString = line.Split(',');                                     

                                    if (valoresString.Length == 645){ 
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
}