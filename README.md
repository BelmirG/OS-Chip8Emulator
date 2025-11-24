# OS-Chip8Emulator
CHIP-8 OS is Python based emulator that leads multiple CHIP-8 ROM files as separate processes. CPU scheduler manages these processes using Round Robin scheduling while dedicated threads handle execution and display rendering

Requirements:
•	Python 3.9 or higher
•	Pyglet library (pip install pyglet)
•	CHIP-8 ROM files
Execution:
python osChip8.py ROM1 ROM2 ROM3
custom quantum (can be adjusted to user needs):
python osChip8.py --quantum=1500 ROM1 ROM2 
