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

#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include "mxc_device.h"
#include "mxc_delay.h"
#include "board.h"
#include "icc.h"
#include "uart.h"
#include "camera.h"
#include "led.h"
#include "dma.h"

#define BAUD_RATE 		115200
#define CAMERA_FREQ		10 * 1000 * 1000
#define ACQ_WIDTH		40
#define ACQ_HEIGHT		40
#define GRAYSCALE_BPP   2

uint32_t process_img()
{
	uint8_t*   raw;
	uint32_t  img_len;
	uint32_t  w, h;

	// We wait for the image to be fully received
	while(!camera_is_image_rcv());

	camera_get_image(&raw, &img_len, &w, &h);

	uint32_t image_mean = 0;

	/* raw contains the starting memory location where the image is stored.
	* Even though the image is grayscale (1 byte per pixel) it occupies a memory
	* location twice as big, where each pixel is stored in the even indices.
	* For this reason, we skip odd numbered index by multiplying for 2 (GRAYSCALE_BBP)
	* while computing the mean, and divide the image length by two before applying it 
	* to the sum of all pixels
	*/
	for(int32_t j=0; j < h; j++) {
		for(int32_t i=0; i < w; i++) {
			uint32_t row_offset = j * w * GRAYSCALE_BPP;
			uint32_t column_offset = i * GRAYSCALE_BPP;
			image_mean += raw[row_offset + column_offset];
		}
	}

	img_len /= 2;
	image_mean /= img_len;
	return image_mean;
}

int main(void)
{
	uint32_t error = 0;
	MXC_ICC_Enable(MXC_ICC0); // Enable cache

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

	// WARNING: using a resolution with an aspect ratio different from 4:3 will distort the image, in our case it's irrelevant
	if((error = camera_setup(ACQ_WIDTH, ACQ_HEIGHT, PIXFORMAT_GRAYSCALE, FIFO_FOUR_BYTE, USE_DMA, dma_channel)) != STATUS_OK) {
		printf("Error in the camera setup: %d\n", error);
		return error;
	}

	MXC_Delay(SEC(2)); // Let debugger interrupt if needed

	uint32_t previous_image_mean = 255;
	while(1) {
		camera_start_capture_image();
		uint32_t image_mean = process_img();

		// Increase the threshold by five to be less subject to the camera noise
		if(image_mean >= (previous_image_mean + 5)) {
			LED_On(LED1);
		} else {
			LED_Off(LED1);
		}

		previous_image_mean = image_mean;
	}
	return 0;
}
