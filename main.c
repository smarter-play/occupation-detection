/******************************************************************************
* MIT License
* 
* Copyright (c) 2022 Deep Vision Consulting s.r.l (https://deepvisionconsulting.com/)
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
 ******************************************************************************/

// CNN_people_presence
// Created using ./ai8xize.py --verbose --test-dir demos --prefix CNN_people_presence --checkpoint-file ../../../people_presence_q8.pth.tar --config-file networks/faceid.yaml --device MAX78000 --board-name EvKit_V1 --display-checkpoint --compact-data --fifo --mexpress --overwrite --timer 0

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "mxc_device.h"
#include "mxc_delay.h"
#include "board.h"
#include "icc.h"
#include "uart.h"
#include "cnn.h"
#include "camera.h"
#include "led.h"
#include "dma.h"

#define OUT_PORT MXC_GPIO3
#define OUT_PIN MXC_GPIO_PIN_1

#define BAUD_RATE 		115200
#define CAMERA_FREQ		10 * 1000 * 1000
// Dimension of camera acquisition resolution
#define ACQ_WIDTH		240
#define ACQ_HEIGHT		180
// Dimension of input of the CNN
#define IMG_WIDTH		120
#define IMG_HEIGHT		160
#define RGB565_BPP		2
// 75% confidence to activate
#define CONFIDENCE		0.75

#define ML_SIZE			(CNN_NUM_OUTPUTS + 3) / 4
#define INPUT_SIZE		3 * IMG_WIDTH * IMG_HEIGHT

volatile uint32_t cnn_time; // Stopwatch
static int32_t ml_data32[ML_SIZE];

typedef struct {
	uint32_t w;
	uint32_t h;
	uint8_t *data;
} rgb565_img;

// Data input: HWC 3x160x120 (57600 bytes total / 19200 bytes per channel):
void load_input(uint32_t bgr_pixel)
{
	while (((*((volatile uint32_t *) 0x50000004) & 1)) != 0); 	// Wait for FIFO 0
	*((volatile uint32_t *) 0x50000008) = bgr_pixel; 			// Write FIFO 0
}

/* The model has been trained using the faceID network, where each output
* was set to either -1 (no people) or 1 (people).
* Since the original net provided 512 uint8_t values, we add all outputs
* and take the average and confront it with a threshold, dependent on the
* level of confidence required for activating.
*/
int calculate_result()
{
	float result = 0.0f;
	for(uint32_t i=0; i<ML_SIZE; i++) {
		// We extract all 8 bit integers from out uint32_t output
		result += (int8_t) (ml_data32[i]);
		result += (int8_t) (ml_data32[i] >> 8);
		result += (int8_t) (ml_data32[i] >> 16);
		result += (int8_t) (ml_data32[i] >> 24);
	}

	result /= ML_SIZE * 4;		// We calculate the mean value of all outputs
	const float threshold = 256 * CONFIDENCE - 128;
	return result > threshold;
}

void make_inference()
{
	while (cnn_time == 0)
	__WFI(); // Wait for CNN

	cnn_unload((uint32_t *) ml_data32);
	cnn_stop();
}

uint8_t *rgb565_get_value(rgb565_img *img, uint32_t x, uint32_t y)
{
	return &img->data[(x + img->w * y) * RGB565_BPP];
}

void rgb565_set_value(rgb565_img *img, uint32_t x, uint32_t y, uint8_t *val)
{
	memcpy(rgb565_get_value(img, x, y), val, RGB565_BPP);
}

uint8_t rgb565_convert_to_gray(uint8_t *data)
{
	uint8_t ur, ug, ub;
	ur = data[0] & 0xF8;
	ug = (data[0] << 5) | ((data[1] & 0xE0) >> 3);
	ub = (data[1] & 0x1F) << 3;

	// Converting to gray scale image according to the ITU-R 601-2 luma transform rounded up
	uint8_t gray = (((float)ur*299/1000) + ((float)ug*587/1000) + ((float)ub*114/1000)) + 0.5f;

	return gray;
}

void process_img()
{
	uint8_t*   raw;
	uint32_t  img_len;
	uint32_t  w, h;

	// We wait for the image to be fully received
	while(!camera_is_image_rcv());

	camera_get_image(&raw, &img_len, &w, &h);

	rgb565_img img = {w, h, raw};

	uint32_t starting_row  = (h - IMG_HEIGHT) / 2;
	uint32_t ending_row = starting_row + IMG_HEIGHT;
	uint32_t starting_column = (w - IMG_WIDTH) / 2;
	uint32_t ending_column = starting_column + IMG_WIDTH;

	cnn_start(); 		// Start CNN processing

	uint32_t counter = 0;

	for(int32_t j=0; j<img.h; j++) {
		for(int32_t i=0; i<img.w; i++) {

			const bool row_inside_crop = j >= starting_row && j < ending_row;
			const bool col_inside_crop = i >= starting_column && i < ending_column;
			const bool pixel_inside_crop = row_inside_crop & col_inside_crop;
			if(!pixel_inside_crop) {
				continue;
			}

			/* These preprocessing steps are done according to the format of the input data
			* during the training process
			*/
			uint8_t gray = rgb565_convert_to_gray(rgb565_get_value(&img, i, j));
			int8_t gr = gray - 128;
			uint32_t gray_preproc = 0x00FFFFFF & ((uint8_t)gr << 16 | (uint8_t)gr << 8 | (uint8_t)gr);
			load_input(gray_preproc);

			counter++;
		}
	}
	make_inference();
}

int main(void)
{
	uint32_t error = 0;
	MXC_ICC_Enable(MXC_ICC0); // Enable cache
	mxc_gpio_cfg_t gpio_out;

	gpio_out.port = OUT_PORT;
    gpio_out.mask = OUT_PIN;
    gpio_out.pad = MXC_GPIO_PAD_NONE;
    gpio_out.func = MXC_GPIO_FUNC_OUT;
    MXC_GPIO_Config(&gpio_out);

	// Switch to 100 MHz clock
	MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
	SystemCoreClockUpdate();

	// We initialize the console UART
	// We can use printf to send serial comunication to the PC using USB
	// Make sure to use the \n character at the end of the message and to use the same baud rate
	// on the serial monitor
	if((error = MXC_UART_Init(MXC_UART_GET_UART(CONSOLE_UART), BAUD_RATE, MXC_UART_IBRO_CLK)) != E_NO_ERROR) {
		printf("Error initializing UART: %d\n", error);
		return error;
	}

	// Initialize DMA for camera interface
	MXC_DMA_Init();
	int dma_channel = MXC_DMA_AcquireChannel();

	// Initialize camera
	if((error = camera_init(CAMERA_FREQ)) != STATUS_OK) {
	  printf("Error initializing camera: %d\n", error);
	  return error;
	}

	/* The acquisition dimension has been chosen to be small enough to fit in SRAM (240x180x2 = 86400 < 128KB) while
	* keeping the 4:3 aspect ratio, and at the same time be big enough to fit inside the CNN input (160x120)
	*/
	if((error = camera_setup(ACQ_WIDTH, ACQ_HEIGHT, PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA, dma_channel)) != STATUS_OK) {
		printf("Error in the camera setup: %d\n", error);
		return error;
	}

	MXC_Delay(SEC(2)); // Let debugger interrupt if needed

	// Enable peripheral, enable CNN interrupt, turn on CNN clock
	// CNN clock: 50 MHz div 1
	cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);

	cnn_init(); // Bring state machine into consistent state
	cnn_load_weights(); // Load kernels
	cnn_load_bias(); // Not used in this network
	cnn_configure(); // Configure state machine

	SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0

	printf("Starting!\n");

	while(1) {
		camera_start_capture_image();
		process_img();

		if(cnn_time != 0) {
			if(calculate_result() > 0) {
				LED_On(LED1);
				MXC_GPIO_OutSet(OUT_PORT, OUT_PIN);
			} else {
				LED_Off(LED1);
				MXC_GPIO_OutClr(OUT_PORT, OUT_PIN);
			}
			cnn_time = 0;
		}
	}
	cnn_disable(); // Shut down CNN clock, disable peripheral
	return 0;
}

/*
  SUMMARY OF OPS
  Hardware: 56,295,040 ops (55,234,560 macc; 1,052,800 comp; 7,680 add; 0 mul; 0 bitwise)
    Layer 0: 8,601,600 ops (8,294,400 macc; 307,200 comp; 0 add; 0 mul; 0 bitwise)
    Layer 1: 22,579,200 ops (22,118,400 macc; 460,800 comp; 0 add; 0 mul; 0 bitwise)
    Layer 2: 11,251,200 ops (11,059,200 macc; 192,000 comp; 0 add; 0 mul; 0 bitwise)
    Layer 3: 5,587,200 ops (5,529,600 macc; 57,600 comp; 0 add; 0 mul; 0 bitwise)
    Layer 4: 2,602,880 ops (2,580,480 macc; 22,400 comp; 0 add; 0 mul; 0 bitwise)
    Layer 5: 2,584,960 ops (2,580,480 macc; 4,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 6: 2,584,960 ops (2,580,480 macc; 4,480 comp; 0 add; 0 mul; 0 bitwise)
    Layer 7: 495,360 ops (491,520 macc; 3,840 comp; 0 add; 0 mul; 0 bitwise)
    Layer 8: 7,680 ops (0 macc; 0 comp; 7,680 add; 0 mul; 0 bitwise)

  RESOURCE USAGE
  Weight memory: 176,048 bytes out of 442,368 bytes total (40%)
  Bias memory:   0 bytes out of 2,048 bytes total (0%)
*/

