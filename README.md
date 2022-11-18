# fracge_gui_3d

To build:
`cargo run --release --all-features`

## My WIP little inefficient and ok DE fractal ray tracer (could be better ofc lol)
### `This was not designed to be perfectly user friendly but just for me to play around with`

-- --
![](https://cdn.discordapp.com/attachments/896508127778832384/1043257118465196162/image.png)
![](https://cdn.discordapp.com/attachments/861404116860796948/1040391944339206154/image.png)
![](https://cdn.discordapp.com/attachments/861404116860796948/1040388575373570219/image.png)
-- --
# IMPORTANT

## You need a controller to move around
There are 2 movement modes

### Using left joystick:
* (DEFAULT) 1: Relative camera move
    * up/down: move in direction of camera forward vector
    * left/right: pan left/right from camera left/right

* 2: relative viewport
    * up/down: move in direction of camera up vector
    * left/right: move in direction of camera left/right

### Using Right joystick:
* (DEFAULT) 1: Relative camera look
    * up/down, rotate around axis of camera left (down literally points down relative to the actual viewport view :3)
    * left/right, rotate around axis of camera up (right literally points right relative to the actual viewport view :3)

* 2: Relative viewport
    * up/down, same as above
    * left/right: rotate around the camera forward axis (literally rotates relative to the viewport view, so you can rotate the * picture)
-- --
## Some info about input grid:

If you see
`0.22|0.001|119.45|4.98|0.1|0.03|8|10|1.7|1|6|13.77|1.31|-2.52|-0.7|0|2.5|4.5|5|12.6|-8|1|1|1|2401.15|12800|-5|1|1|3.72|1|1|10|60|1|1|1|-500|0|0|1.01|1|1|0.987|1|1|0|2|1.98|0.9|2|0.25|3|2.25|49.81|1.62|0.55|-2.7|3.45|2|2|2|600|1201.66|`
at the end of your txt generated in your output folder, those are your input grid params!

`all you have to do is copy just that line, paste it into that text box below the input grid, and then hit "Accept" :3`

I will refer to these with the in-program notation:
> ### r4[0] refers to the FIRST ELEMENT in the FOURTH ROW
> ### r0[0] - r0[2] This refers to the range of the elements from 0 to 2 of the first row
> ### r0[1] - r2[1] This refers to the range from 0 to 3 of the first three rows, the second element
&nbsp;
* ### r7[0] - r7[2] : `LIGHT POSITION`
&nbsp;
* ### r0[4] : `AO BRIGHTNESS (GUH?)`
* ### r0[5] : `AO SCALE`
* ### r0[6] : `AO STEPS`
&nbsp;
* ### r5[5] : `Camera Depth Of Field `
* ### r5[6] : `Camera Aperature`
    * Keep this at 0 for none, you need a TON of samples for smoothness, sorry :(
* ### r0[7] : `Focal Distance`
&nbsp;
* ### r1[3] : `Color Scale`
* ### r1[4] : `Color Exponent`
* ### r1[5]-r1[6]-r1[7] : `Color hue adjustments, you probably want to change r1[7]`
&nbsp;

# Wtf is wrong with the output?
## Please disable the two checkmarks which signify the KEEP RATIO, theyre really buggy
&nbsp;

## You ALSO need to ensure the output split grid size is some multiple of your output dimensions

&nbsp;

## SOMETIMES, it messes up and a change in your viewport ratio will lead to a change in the export ratio, and you'll need to re-enter the export ratio :( 

&nbsp;


![](https://cdn.discordapp.com/attachments/861404116860796948/1043265810493349888/image.png)

Here you can see `S:` `X:` `Y:` `Y%:` `VP_samples/p:` `X/Y CHUNKS`

* ### `S`: Samples per pass that the GPU does, keep this low to free up resource and not slow down your pc
&nbsp;
* ### `X` & `Y`: The GPU splits up the rendering into smaller chunks, these are the dimensions of each of those chunks. If you see decimals on `X/Y CHUNKS`, please update it to a whole integer divisor or you will experience issues :(
&nbsp;
* ### `Y%`: How many chunks to render in a row before the GPU stops for a tiny tiny bit of time (its not a wait, its just moving data from the buffer and then continuing, it can help with system instability during intense rendering IG, but I'm sure as **** no expert on GPU's yet hahaha)
&nbsp;
* ### `VP_samples/p`: Samples rendered per pass in viewport
&nbsp;
* ### `X/Y CHUNKS`: Calculates expected chunks from input X/Y from above
&nbsp;
-- --
&nbsp;

* # Questions?
&nbsp;

# What about SPEED?
## There are speed adjustments in the axis tab

* ## You can also hold down the left trigger to slow down the movement
* ## You can also hold down the right trigger to slow down the camera rotations
&nbsp;

# How do I make a fractal?
* ## You need to program it in within the KERNEL tab, please be careful with loops, I am not responsible for system crashes
* ## I have yet to document what the absolute F the input grid does, that is a LOT of work, coming soon (tm)
&nbsp;

# Whats wrong with the shading?
## Sometimes the rendering glitches out, fix your iter count or something lol
&nbsp;

# Why does it crash with inputs?
## I need to go back over EVERY single input box and add in more checks to prevent it from panicing 
* A tedious thing to do atm, I havent had much of a problem
