# Smarter Play Occupation Detection

Code is an adapted version of the Deep Vision Consulting People Detection example, which uses the standard Maxim Integrated Face ID model trained using images taken from the COCO object detection dataset.

## Board connections

![](boardschem.png)

The board can get (5V) power from its USB port or from the VBUS pin on the schematic.

The red built-in LED and the detection output signal on pin P3_1 (third on the 12-pin row starting from the black line-out audio connector (opposite the USB connector, on the side of the white battery connector)) stays HIGH (at around 3.3V) while the board is detecting people (by taking pictures with its builtin camera and passing them through a CNN).
