Disclaimer
========================
Please note that this project was started in 2014, when I had almost no work experience.   
I hope to come back one day and refactor the hell out of it:)   
But Fluid simulation on GPU is still one of the most complex things I've done as a home project, so I will show it off:)

About
========================
This is a realtime fluid simulation executed on GPU using CUDA.   
Simulation is based on a paper about [smooth particle hydrodynamics](http://mmacklin.com/pbf_sig_preprint.pdf) (by Miles Macklin, Matthias Muller: "Position Based Fluids", 2013)

Watch
========================
[![Demonstration video](https://j.gifs.com/wVMXjw.gif)](https://www.youtube.com/watch?v=P9LP9VxWFE8)

Usage
========================
To run you would need an 
- Nvidia GPU with computing capability at least 2.0 (most graphics cards in the past 5 years)
- Windows 64 bit
- Opengl

To compile this project you would need VS 2013 and CUDA Computing Toolkit V7.5

Hotkeys
========================
- Use arrows to control camera rotation
- W/A/S/D to control gravity
- E/R Decrease/Increase viscosity
- T/Y Decrease/Increae solver iteration count
- Space to drop another bunch of particles
