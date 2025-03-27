/*******************************************************************************
* File Name:   main.c
*
* Description: This project is the SPI master for communications with an AFE ADS1299

********************************************************************************/

/*******************************************************************************
* Header Files
*******************************************************************************/
#pragma once
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

/******************************************************************************
* Macros                                                                     */
/*****************************************************************************/
/* SPI baud rate in Hz */
#define SPI_FREQ_HZ                (3623000UL)
/* SPI transfer bits per frame */
#define BITS_PER_FRAME             (8)

/* Convertion from 3 bytes info to uint32_t */
#define CONVERT(C) 	(((uint32_t)info.data.ch[C].Channel[0]) << 16) + (((uint32_t)info.data.ch[C].Channel[1]) << 8) + ((uint32_t)info.data.ch[C].Channel[2])

/* Number of channels to read */
#define NCHANNELS 5

/* Number of characteristics to the model */
#define NCHARACS 4

/* 8.1us delay */
#define COMMON_DELAY               (75) // 10 ~= 0.5 us
/* 11.5us delay */
#define LARGE_DELAY                (850)

/* Chip select */
#define CS_PIN                     (P6_0)
#define DRDY_PIN                   (P6_1)

/* Butterworth filter parameters */
#define NZEROS 7
#define NPOLES 7

/* FFT variables */
#define FFT_BUFFER_SIZE 256
#define SAMPLE_RATE 250

/* ADS1299 register addresses */
#define ADS1299_REG_DEVID         (0x0000u)
#define ADS1299_REG_CONFIG1       (0x0001u)
#define ADS1299_REG_CONFIG2       (0x0002u)
#define ADS1299_REG_CONFIG3       (0x0003u)
#define ADS1299_REG_LOFF          (0x0004u)
#define ADS1299_REG_CH1SET        (0x0005u)
#define ADS1299_REG_CH2SET        (0x0006u)
#define ADS1299_REG_CH3SET        (0x0007u)
#define ADS1299_REG_CH4SET        (0x0008u)
#define ADS1299_REG_CH5SET        (0x0009u)
#define ADS1299_REG_CH6SET        (0x000Au)
#define ADS1299_REG_CH7SET        (0x000Bu)
#define ADS1299_REG_CH8SET        (0x000Cu)
#define ADS1299_REG_BIASSENSP     (0x000Du)
#define ADS1299_REG_BIASSENSN     (0x000Eu)
#define ADS1299_REG_LOFFSENSP     (0x000Fu)
#define ADS1299_REG_LOFFSENSN     (0x0010u)
#define ADS1299_REG_LOFFFLIP      (0x0011u)
#define ADS1299_REG_LOFFSTATP     (0x0012u)
#define ADS1299_REG_LOFFSTATN     (0x0013u)
#define ADS1299_REG_GPIO          (0x0014u)
#define ADS1299_REG_MISC1         (0x0015u)
#define ADS1299_REG_MISC2         (0x0016u)
#define ADS1299_REG_CONFIG4       (0x0017u)

uint8_t Conf_addresses[] = {ADS1299_REG_CONFIG1, ADS1299_REG_CONFIG2, ADS1299_REG_CONFIG3,
		ADS1299_REG_LOFF,
		ADS1299_REG_CH1SET, ADS1299_REG_CH2SET, ADS1299_REG_CH3SET, ADS1299_REG_CH4SET, ADS1299_REG_CH5SET, ADS1299_REG_CH6SET,	ADS1299_REG_CH7SET,	ADS1299_REG_CH8SET,
		ADS1299_REG_BIASSENSP, ADS1299_REG_BIASSENSN,
		ADS1299_REG_LOFFSENSP, ADS1299_REG_LOFFSENSN,
		ADS1299_REG_LOFFFLIP,
		ADS1299_REG_GPIO,
		ADS1299_REG_MISC1, ADS1299_REG_MISC2,
		ADS1299_REG_CONFIG4
};
/* Only-reading registers */
uint8_t Read_addresses[] = {ADS1299_REG_DEVID, ADS1299_REG_LOFFSTATP, ADS1299_REG_LOFFSTATN};
/* Default only-reading values */
uint8_t Def_values[] = {0x3E,0x00,0x00};
/* Configuration values */
uint8_t Conf_values[] = {0x96,0xD0,0xEC, //CONFIG 1,2,3 (El config3 no está probado)
		0x00, //LOFF
		0x60,0x60,0x60,0x60,0x60,0x66,0x68,0xE1, //CHSET (0xE1 -> Apagado ; 0x65 -> Señal de prueba ; 0x60 -> Electrodo normal)
		0x1F,0x1F, //BIASSENS (Para los 5 canales, poner 1F - 01, para pruebas con un solo canal, 01 - 01)0x07,0x07, //BIASSENS
		0x00,0x00, //LOFFSENS
		0x00, //LOFFFLIP
		0x00, //GPIO
		0x20,0x00, //MISC
		0x00, //CONFIG4
};

/******************************************************************************
* Global variables
*******************************************************************************/
/* SPI object */
cyhal_spi_t mSPI;
/* Error handler */
cy_rslt_t result;
/* Data ready interrupt variable */
bool Interrupt_Occurred = false;
/* Channels to read from ADS1299*/
size_t chs2Read;
/* Real converted data */
double valoresReales[NCHANNELS];
/* Raw data from ADS1299 */
int32_t CHS[NCHANNELS];

/* FFT variables */
float input_fft[NCHANNELS][FFT_BUFFER_SIZE];
float output_fft[NCHANNELS][FFT_BUFFER_SIZE];
int fft_index[NCHANNELS];
float fft_value[NCHANNELS][FFT_BUFFER_SIZE/2];
bool  fft_ready[NCHANNELS] = {false};

/* Signal characteristics for SVM model */
//double mu_power[NCHANNELS];
//double beta_power[NCHANNELS];
//double avg_freq[NCHANNELS];
//double rms_value[NCHANNELS];
double characs[NCHANNELS][NCHARACS];
double sum_val[NCHANNELS] = {0.0};
double total_power[NCHANNELS];
double weighted_sum[NCHANNELS];
double power;

/* Frequency bins */
double bins[FFT_BUFFER_SIZE/2];

/* Initialize FFT instance */
arm_rfft_fast_instance_f32 fft_instance[NCHANNELS];

/* Verification counter */
int count = 0;

/* Structure to store raw data */
union {
	struct DATOS {
		uint8_t status[3];
		struct CH {
			uint8_t Channel[3];
		} ch[8];
	} data;
	uint8_t bytes[27];
} info;

/* Commands definition */
typedef enum {
    WAKEUP = 0x02,
    STANDBY = 0x04,
    RESET = 0x06,
    START = 0x08,
    STOP = 0x0A,
	RDATAC = 0x10,
	SDATAC = 0x11,
	RDATA = 0x12,
} Commands;

/* Butterworth filter matrixes declaration */
static float xv_LP[NCHANNELS][NZEROS+1], yv_LP[NCHANNELS][NPOLES+1],xv_HP[NCHANNELS][NZEROS+1], yv_HP[NCHANNELS][NPOLES+1], xv_BP[NCHANNELS][2*NZEROS+1], yv_BP[NCHANNELS][2*NZEROS+1];

/******************************************************************************
* FUNCTION DEFINITION                                                         */
/*******************************************************************************/
/*******************************************************************************
* Function Name: delay_clks
********************************************************************************
* Summary:
* waits for clks clock times
*
* Parameters:
*  uint32_t clks - number of clock times
*
* Return:
*  void
*
*******************************************************************************/
void delay_clks(uint32_t clks) {
    while(clks > 0) {
        __asm__("nop");
        clks--;
    }
}
/*******************************************************************************
* Function Name: handle_error
********************************************************************************
* Summary:
* User defined error handling function. If any init failed, stops the execution
*
* Parameters:
*  uint32_t status - status indicates success or failure
*
* Return:
*  void
*
*******************************************************************************/
void handle_error(uint32_t status){
    if (status != CY_RSLT_SUCCESS){
        CY_ASSERT(0);
    }
}

/*******************************************************************************
* Function Name: SPI_ReadReg_32 (RREG)
********************************************************************************
* Summary:
*  Read registers from AFE
*
* Parameters:
*  uint8_t address - Reading starting address
*  uint8_t *datain - Pointer to memory space for the retreived data
*
* Return:
*  Store registers in datain as uint32_t
*
*******************************************************************************/
void SPI_ReadReg_32(uint8_t address, uint32_t *datain) {
	uint8_t txvect[] = {(address & 0x1f) | 0x20, 0x00, 0x00};
	result = cyhal_spi_transfer(&mSPI, txvect, 3, datain, 3, 0);
	handle_error(result);

}

/*******************************************************************************
* Function Name: SPI_ReadReg (RREG)
********************************************************************************
* Summary:
*  Make the whole process reading a register and extracting the value in a uint8_t
*
* Parameters:
*  uint8_t address - Reading starting address
*
* Return:
*  uint8_t - register value
*
*******************************************************************************/
uint8_t SPI_ReadReg(uint8_t address){
	uint32_t Pre_reg;
	SPI_ReadReg_32(address, &Pre_reg);
	uint8_t Readed_Reg = (Pre_reg >> 16) & 0xFF;
	return Readed_Reg;
}

/*******************************************************************************
* Function Name: SPI_WriteReg (WREG)
********************************************************************************
* Summary:
*  Writes registers to the AFE
*
* Parameters:
*  uint8_t address - Writing starting address
*  uint8_t length - Number of registers/bytes to write
*  uint8_t *datain - Pointer to memory space of the data to write
*
* Return:
*  void
*
*******************************************************************************/
void SPI_WriteReg(uint8_t address, uint8_t dataout) {
	uint8_t txvect[] = {(address & 0x1f) | 0x40, 0x00, dataout};
	result = cyhal_spi_transfer(&mSPI, txvect, 3, NULL, 0, 0);
	handle_error(result);
}

/*******************************************************************************
* Function Name: SPI_ReadAll
********************************************************************************
* Summary:
*  Read the information of the 8 channels of the AFE
*
* Parameters:
*  void
*
* Return:
*  Write the data into
*
*******************************************************************************/
void SPI_ReadAll(void) {
	/* Select the device to read DOUT*/
	cyhal_gpio_write(CS_PIN, 0);
	/* Reads DOUT */
	memset(&info, 0, sizeof(info));
	result = cyhal_spi_transfer(&mSPI, NULL, 0, info.bytes, 27, 0);
	handle_error(result);
	/*Reset SPI interface */
	cyhal_gpio_write(CS_PIN, 1);
}

/*******************************************************************************
* Function Name: IssueResetPulse
********************************************************************************
* Summary:
*  Sends a RESET command at demand
*
* Parameters:
*  void
*
* Return:
*  void
*
*******************************************************************************/
void SPI_SendCommand(Commands cmd){
	result = cyhal_spi_send(&mSPI, cmd);
	handle_error(result);
	if (cmd == RESET){
		delay_clks(18);
	}
}

/*******************************************************************************
* Function Name: ProcessValues
********************************************************************************
* Summary:
*  Convert the data from the ADS1299 (24 bits) into the real value (double) and filter it
*  for a better after analysis
*
* Parameters:
*  size_t chs2Read - Channels to read
*
* Return:
*  double* - Real values (Volts) readed from ADS1299
*
*******************************************************************************/
void ProcessValues(double* valoresReales, size_t chs2Read, int32_t* CHS) {
	/* Process the twos complement data */
    for (int i = 0; i < NCHANNELS;i++) {
        if (CHS[i] & 0x800000) {
            CHS[i] |= 0xFF000000;
        }
        /* Scales data and stores it as double for a correct voltage value */
        valoresReales[i] = CHS[i] * 0.9e-8;

        /* Filtering data */

        //Web page//
        //band-stop filter (50Hz-70Hz @250SPS)
        xv_BP[i][0] = xv_BP[i][1]; xv_BP[i][1] = xv_BP[i][2]; xv_BP[i][2] = xv_BP[i][3]; xv_BP[i][3] = xv_BP[i][4]; xv_BP[i][4] = xv_BP[i][5]; xv_BP[i][5] = xv_BP[i][6]; xv_BP[i][6] = xv_BP[i][7]; xv_BP[i][7] = xv_BP[i][8]; xv_BP[i][8] = xv_BP[i][9]; xv_BP[i][9] = xv_BP[i][10]; xv_BP[i][10] = xv_BP[i][11]; xv_BP[i][11] = xv_BP[i][12]; xv_BP[i][12] = xv_BP[i][13]; xv_BP[i][13] = xv_BP[i][14];
		xv_BP[i][14] = valoresReales[i];
		yv_BP[i][0] = yv_BP[i][1]; yv_BP[i][1] = yv_BP[i][2]; yv_BP[i][2] = yv_BP[i][3]; yv_BP[i][3] = yv_BP[i][4]; yv_BP[i][4] = yv_BP[i][5]; yv_BP[i][5] = yv_BP[i][6]; yv_BP[i][6] = yv_BP[i][7]; yv_BP[i][7] = yv_BP[i][8]; yv_BP[i][8] = yv_BP[i][9]; yv_BP[i][9] = yv_BP[i][10]; yv_BP[i][10] = yv_BP[i][11]; yv_BP[i][11] = yv_BP[i][12]; yv_BP[i][12] = yv_BP[i][13]; yv_BP[i][13] = yv_BP[i][14];
		yv_BP[i][14] =   (xv_BP[i][0] + xv_BP[i][14]) -   0.9075805865 * (xv_BP[i][1] + xv_BP[i][13]) +   7.3530153661 * (xv_BP[i][2] + xv_BP[i][12])
					 -   5.5217668266 * (xv_BP[i][3] + xv_BP[i][11]) +  22.7749672950 * (xv_BP[i][4] + xv_BP[i][10]) -  13.9196114330 * (xv_BP[i][5] + xv_BP[i][9])
					 +  38.5598583060 * (xv_BP[i][6] + xv_BP[i][8]) -  18.6108510020 * xv_BP[i][7]
					 + ( -0.1012581378 * yv_BP[i][0]) + (  0.1067072919 * yv_BP[i][1])
					 + ( -0.9856096920 * yv_BP[i][2]) + (  0.8648848442 * yv_BP[i][3])
					 + ( -4.0911642588 * yv_BP[i][4]) + (  2.9435892260 * yv_BP[i][5])
					 + ( -9.4114517573 * yv_BP[i][6]) + (  5.3942315447 * yv_BP[i][7])
					 + (-12.9886022980 * yv_BP[i][8]) + (  5.6260667326 * yv_BP[i][9])
					 + (-10.7789857630 * yv_BP[i][10]) + (  3.1758667365 * yv_BP[i][11])
					 + ( -4.9937808523 * yv_BP[i][12]) + (  0.7613467028 * yv_BP[i][13]);
		double BS_signal = yv_BP[i][14];

		//Low-pass filter (35Hz @ 250SPS)
		xv_LP[i][0] = xv_LP[i][1]; xv_LP[i][1] = xv_LP[i][2]; xv_LP[i][2] = xv_LP[i][3]; xv_LP[i][3] = xv_LP[i][4]; xv_LP[i][4] = xv_LP[i][5]; xv_LP[i][5] = xv_LP[i][6]; xv_LP[i][6] = xv_LP[i][7];
		xv_LP[i][7] = BS_signal;
		yv_LP[i][0] = yv_LP[i][1]; yv_LP[i][1] = yv_LP[i][2]; yv_LP[i][2] = yv_LP[i][3]; yv_LP[i][3] = yv_LP[i][4]; yv_LP[i][4] = yv_LP[i][5]; yv_LP[i][5] = yv_LP[i][6]; yv_LP[i][6] = yv_LP[i][7];
		yv_LP[i][7] =   (xv_LP[i][0] + xv_LP[i][7]) + 7 * (xv_LP[i][1] + xv_LP[i][6]) + 21 * (xv_LP[i][2] + xv_LP[i][5])
					 + 35 * (xv_LP[i][3] + xv_LP[i][4])
					 + (  0.0161309544 * yv_LP[i][0]) + ( -0.1764537312 * yv_LP[i][1])
					 + (  0.8531799244 * yv_LP[i][2]) + ( -2.3781565370 * yv_LP[i][3])
					 + (  4.1558558031 * yv_LP[i][4]) + ( -4.6155169516 * yv_LP[i][5])
					 + (  3.0619031292 * yv_LP[i][6]);
		double LP_signal = yv_LP[i][7];

		//High-pass filter (7Hz @ 250SPS)
        xv_HP[i][0] = xv_HP[i][1]; xv_HP[i][1] = xv_HP[i][2]; xv_HP[i][2] = xv_HP[i][3]; xv_HP[i][3] = xv_HP[i][4]; xv_HP[i][4] = xv_HP[i][5]; xv_HP[i][5] = xv_HP[i][6]; xv_HP[i][6] = xv_HP[i][7];
        xv_HP[i][7] = LP_signal;
        yv_HP[i][0] = yv_HP[i][1]; yv_HP[i][1] = yv_HP[i][2]; yv_HP[i][2] = yv_HP[i][3]; yv_HP[i][3] = yv_HP[i][4]; yv_HP[i][4] = yv_HP[i][5]; yv_HP[i][5] = yv_HP[i][6]; yv_HP[i][6] = yv_HP[i][7];
        yv_HP[i][7] =   (xv_HP[i][7] - xv_HP[i][0]) + 7 * (xv_HP[i][1] - xv_HP[i][6]) + 21 * (xv_HP[i][5] - xv_HP[i][2])
        			 + 35 * (xv_HP[i][3] - xv_HP[i][4])
        			 + (  0.4529683403 * yv_HP[i][0]) + ( -3.5288387808 * yv_HP[i][1])
        			 + ( 11.8041886730 * yv_HP[i][2]) + (-21.9798148520 * yv_HP[i][3])
        			 + ( 24.6071872740 * yv_HP[i][4]) + (-16.5652186770 * yv_HP[i][5])
        			 + (  6.2095244481 * yv_HP[i][6]);
        valoresReales[i] = yv_HP[i][7];

        //Signal storage
        input_fft[i][fft_index[i]] = valoresReales[i];
        sum_val[i] += valoresReales[i]*valoresReales[i];
        fft_index[i]++;

        //FFT generation and characteristics calculation
        if(fft_index[i] >= FFT_BUFFER_SIZE){
        	arm_rfft_fast_f32(&fft_instance[i], &input_fft[i], &output_fft[i], 0);
        	for(int j = 0; j < FFT_BUFFER_SIZE; j += 2){
        		fft_value[i][j] = sqrtf((output_fft[i][j]*output_fft[i][j])+(output_fft[i][j+1]*output_fft[i][j+1]));
        	}

        	// SVM model characteristics calculation
        	/* Mu power */
        	for (int j = 8; j <= 12; j++) {
        		characs[i][0] += fft_value[i][j]*fft_value[i][j];
			}

        	/* Beta power */
			for (int j = 13; j <= 32; j++) {
				characs[i][1] += fft_value[i][j]*fft_value[i][j];
			}

			/* Average frequency */
			for (int j = 0; j < FFT_BUFFER_SIZE/2; j++) {
				power = fft_value[i][j]*fft_value[i][j];
				weighted_sum[i] += bins[j]*power;
				total_power[i] += power;
			}
			characs[i][2] = (total_power[i] != 0.0) ? (weighted_sum[i] / total_power[i]) : 0.0;

			/* RMS value */
		    characs[i][3] = sqrt(sum_val[i]/FFT_BUFFER_SIZE);

		    /* Reinitialize values */
        	fft_index[i] = 0;
        	fft_ready[i] = true;
        }

    }
}

/*******************************************************************************
* Function Name: Data_Ready_Callback
********************************************************************************
* Summary:
*  Reads the DRDY pin callback to store the new data.
*
* Parameters:
*  void
*
* Return:
*  void
*
*******************************************************************************/
void Data_Ready_Callback (void *callback_arg, cyhal_gpio_event_t event){
	Interrupt_Occurred = true;
}
/*******************************************************************************
* Function Name: allFFTReady
********************************************************************************
* Summary:
*  Verify if at least one element of an array is false
*
* Parameters:
*  boll arr[], int size
*
* Return:
*  void
*
*******************************************************************************/
bool allFFTReady(const bool arr[], int size) {
    for (int i = 0; i < size; i++) {
        if (!arr[i]) {
            return false;
        }
    }
    return true;
}
/*******************************************************************************
* Function Name: main
********************************************************************************
* Summary:
* The main function.
*   1. Initializes the board, retarget-io and SPI interface
*   2. Configures the SPI Master to send command packet to the slave
*
* Parameters:
*  void
*
* Return:
*  void
*
*******************************************************************************/
int main(void){

/*******************************************************************************
* VOID SETUP
********************************************************************************/

	/* Enable interrupts */
	__enable_irq();

	/******************************************** INITIALIZATION ********************************************/
	/* Initialize the device and board peripherals */
    result = cybsp_init();
    handle_error(result);

    /* Initialize retarget-io for uart logs */
    result = cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,4000000);
    handle_error(result);

    /* Initialize blue LED */
	result = cyhal_gpio_init(CYBSP_LED_RGB_BLUE, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, CYBSP_LED_STATE_OFF);
	handle_error(result);

	/* Initialize CS pin */
	result = cyhal_gpio_init(CS_PIN, CYHAL_GPIO_DIR_OUTPUT, CYHAL_GPIO_DRIVE_STRONG, 0);
	handle_error(result);

	/* Initialize DRDY_pin */
	result = cyhal_gpio_init(DRDY_PIN, CYHAL_GPIO_DIR_INPUT, CYHAL_GPIO_DRIVE_PULL_NONE, CYBSP_BTN_OFF);
	handle_error(result);

	/* Initialize SPI interface as master */
	result = cyhal_spi_init(&mSPI,CYBSP_SPI_MOSI,CYBSP_SPI_MISO,CYBSP_SPI_CLK,CYBSP_SPI_CS,NULL,BITS_PER_FRAME,CYHAL_SPI_MODE_01_MSB,false);
	handle_error(result);

	/* Set the SPI baud rate */
	result = cyhal_spi_set_frequency(&mSPI, SPI_FREQ_HZ);
	handle_error(result);

	/* Defining the interruption for DRDY (Data conversion ready) */
	cyhal_gpio_callback_data_t Data_Ready = {
		.callback = Data_Ready_Callback
	};
	cyhal_gpio_register_callback(DRDY_PIN, &Data_Ready);

	/* Initialize fft instances */
	for (int i = 0; i<NCHANNELS; i++){
		arm_rfft_fast_init_f32(&fft_instance[i], FFT_BUFFER_SIZE);
		fft_index[i] = 0;
	}

	/* ANSI ESC sequence for clear screen */
    printf("\x1b[2J\x1b[;H");
    /* Welcoming */
    printf("================================================== \r\n");
    printf("       Welcome to the AFE-PSOC BMI program         \r\n\n");
    printf("                Version: 1.0.0                     \r\n");
    printf("    	     Realased 15/11/2024                   \r\n\n");
    printf("Sebastian Perrone Claro - Nicolas Quijano Macías   \r\n\n");
    printf("               Universidad EIA                     \r\n");
    printf("================================================== \r\n\n");

/******************************************** CONFIGURATION PROCESS ********************************************/
	/* Start SPI CS in LOW */
	cyhal_gpio_write(CS_PIN, 0);

	/* Waiting for voltage stabilization */
	delay_clks(100000);

	/* Reset SPI interface */
	cyhal_gpio_write(CS_PIN, 1);
	delay_clks(10);
	cyhal_gpio_write(CS_PIN, 0);

	/* Waiting for voltage stabilization */
	delay_clks(1000000);

    /* Send SDATAC command */
    SPI_SendCommand(SDATAC);
    delay_clks(COMMON_DELAY);

    /* Send STOP command */
	SPI_SendCommand(STOP);

	/* Verifying SPI Communication with AFE ID */
	uint8_t Readed_ID = SPI_ReadReg(ADS1299_REG_DEVID);
	printf("ID AFE: %02x \r\n\n", Readed_ID);

	if(Readed_ID == Def_values[0]){
		printf("Device found ---> Sending registers \r\n");

		/* Send configuration registers */
		uint8_t all_good = 1;
		for(int i = 0; i < sizeof(Conf_addresses); i++){
			/* Sends register */
			SPI_WriteReg(Conf_addresses[i], Conf_values[i]);
			delay_clks(COMMON_DELAY);

			/* Read register */
			uint8_t reg = SPI_ReadReg(Conf_addresses[i]);

			/* Compares */
			if(Conf_values[i] != reg){
				printf("Problem sending %02x register.... Please reset \r\n", Conf_addresses[i]);
				all_good = 0;
			}
		}

		if(all_good){
			printf("All registers written successfully ---> Starting conversions \r\n\n");
			printf("================================================== \r\n\n");

			/* Send START command */
			SPI_SendCommand(START);
			delay_clks(COMMON_DELAY);

			/* Send RDATAC command */
			SPI_SendCommand(RDATAC);

			/* Enable DRDY interrupts */
			cyhal_gpio_enable_event(DRDY_PIN, CYHAL_GPIO_IRQ_FALL, CYHAL_ISR_PRIORITY_DEFAULT, true);

			/* frequency bins */
			for(int i = 0; i < FFT_BUFFER_SIZE/2; i++){
				bins[i] = (double)(i*SAMPLE_RATE/((float)FFT_BUFFER_SIZE));
			}
/*******************************************************************************
* VOID LOOP
********************************************************************************/
			for (;;){

				if(Interrupt_Occurred){
					count ++;
					/* Reading all channels */
					SPI_ReadAll();

					/* Storing raw data into a 32 bits format*/
					CHS[0] = CONVERT(0);
					CHS[1] = CONVERT(1);
					CHS[2] = CONVERT(2);
				    CHS[3] = CONVERT(3);
				    CHS[4] = CONVERT(4);
				    chs2Read = sizeof(CHS) / sizeof(CHS[0]);

					/* Format and store data in a real number form */
					ProcessValues(valoresReales, chs2Read, CHS);

					/* Signal for a communication start*/
					printf("I\n");
					fflush(stdout);

					/* Transmit data to PC to be stored and plotted*/
					for(int i = 0; i < NCHANNELS; i++){
						printf("%.6f\n", valoresReales[i]);
						fflush(stdout);
					}

					/* Verify if every FFT is ready */
					if(allFFTReady(fft_ready, NCHANNELS)){
						printf("C\n");
						/*Sending SVM characteristics*/
						for (int i = 0; i < NCHANNELS; i++){
							for (int j = 0; j < NCHARACS; j++){
								printf("%.6f\n",characs[i][j]);
								fflush(stdout);
							}
						}
						/*Sending FFT data*/
						printf("F\n");
						for (int i = 0; i < NCHANNELS; i++){
							fft_ready[i] = false;
							for(int j = 0; j < FFT_BUFFER_SIZE/2; j++){
								printf("%.6f\n", fft_value[i][j]);
								fflush(stdout);
							}
						}
						printf("NEW\n");
						fflush(stdout);
						count = 0;
					}else{
						printf("OLD\n");
						fflush(stdout);
					}

					/* Reset the interruption flag */
					Interrupt_Occurred = false;
				}
			}
		}else{
			printf("Error loading registers.... Please reset");
			for (;;){
				cyhal_syspm_deepsleep();
			}
		}
	}else{
		printf("ID Error.... Device not found");
		for (;;){
			cyhal_syspm_deepsleep();
		}
	}
}


/* [] END OF FILE */

