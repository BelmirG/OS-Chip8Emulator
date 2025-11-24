import threading
import time
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import os
import sys

try:
    import pyglet
    from pyglet.gl import *
except ImportError:
    print("ERROR: pyglet not installed!")
    print("Install with: pip install pyglet")
    sys.exit(1)

# Keyboard mapping
KEY_MAP = {
    pyglet.window.key._1: 0x1, pyglet.window.key._2: 0x2,
    pyglet.window.key._3: 0x3, pyglet.window.key._4: 0xc,
    pyglet.window.key.Q: 0x4, pyglet.window.key.W: 0x5,
    pyglet.window.key.E: 0x6, pyglet.window.key.R: 0xd,
    pyglet.window.key.A: 0x7, pyglet.window.key.S: 0x8,
    pyglet.window.key.D: 0x9, pyglet.window.key.F: 0xe,
    pyglet.window.key.Z: 0xa, pyglet.window.key.X: 0,
    pyglet.window.key.C: 0xb, pyglet.window.key.V: 0xf
}

class ProcessState(Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"

class SchedulingAlgorithm(Enum):
    ROUND_ROBIN = "Round Robin"
    FCFS = "First-Come-First-Served"
    PRIORITY = "Priority Scheduling"

@dataclass
class ProcessControlBlock:
    pid: int
    name: str
    state: ProcessState
    priority: int
    program_counter: int
    registers: List[int]
    index_register: int
    stack: List[int]
    memory: List[int]
    display_buffer: List[int]
    delay_timer: int
    sound_timer: int
    key_inputs: List[int]
    burst_time: int
    arrival_time: float
    cpu_time: float = 0
    context_switches: int = 0
    
    def save_context(self, cpu_state):
        self.program_counter = cpu_state['pc']
        self.registers = cpu_state['gpio'].copy()
        self.index_register = cpu_state['index']
        self.stack = cpu_state['stack'].copy()
        self.memory = cpu_state['memory'].copy()
        self.display_buffer = cpu_state['display'].copy()
        self.delay_timer = cpu_state['delay_timer']
        self.sound_timer = cpu_state['sound_timer']
        self.key_inputs = cpu_state['keys'].copy()
        self.context_switches += 1
    
    def restore_context(self):
        return {
            'pc': self.program_counter,
            'gpio': self.registers.copy(),
            'index': self.index_register,
            'stack': self.stack.copy(),
            'memory': self.memory.copy(),
            'display': self.display_buffer.copy(),
            'delay_timer': self.delay_timer,
            'sound_timer': self.sound_timer,
            'keys': self.key_inputs.copy()
        }

class CHIP8Core:
    def __init__(self):
        self.memory = [0] * 4096
        self.gpio = [0] * 16
        self.display_buffer = [0] * 64 * 32
        self.stack = []
        self.key_inputs = [0] * 16
        self.pc = 0x200
        self.index = 0
        self.delay_timer = 0
        self.sound_timer = 0
        self.opcode = 0
        self.should_draw = False
        
        fonts = [
            0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70,
            0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0, 0x10, 0xF0, 0x10, 0xF0,
            0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
            0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40,
            0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0, 0x10, 0xF0,
            0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
            0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0,
            0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80
        ]
        for i, byte in enumerate(fonts):
            self.memory[i] = byte
    
    def load_program(self, program_data):
        for i, byte in enumerate(program_data):
            if i + 0x200 < 4096:
                self.memory[i + 0x200] = byte
    
    def get_state(self):
        return {
            'pc': self.pc, 'gpio': self.gpio.copy(), 'index': self.index,
            'stack': self.stack.copy(), 'memory': self.memory.copy(),
            'display': self.display_buffer.copy(), 'delay_timer': self.delay_timer,
            'sound_timer': self.sound_timer, 'keys': self.key_inputs.copy()
        }
    
    def set_state(self, state):
        self.pc = state['pc']
        self.gpio = state['gpio'].copy()
        self.index = state['index']
        self.stack = state['stack'].copy()
        self.memory = state['memory'].copy()
        self.display_buffer = state['display'].copy()
        self.delay_timer = state['delay_timer']
        self.sound_timer = state['sound_timer']
        self.key_inputs = state['keys'].copy()
    
    def execute_instruction(self):
        if self.pc >= 4094:
            return False
        
        self.opcode = (self.memory[self.pc] << 8) | self.memory[self.pc + 1]
        self.pc += 2
        vx = (self.opcode & 0x0f00) >> 8
        vy = (self.opcode & 0x00f0) >> 4
        
        if self.opcode == 0x00E0:  
            self.display_buffer = [0] * 64 * 32
            self.should_draw = True
        elif self.opcode == 0x00EE:  
            if self.stack:
                self.pc = self.stack.pop()
        elif (self.opcode & 0xF000) == 0x1000:
            self.pc = self.opcode & 0x0FFF
        elif (self.opcode & 0xF000) == 0x2000:
            self.stack.append(self.pc)
            self.pc = self.opcode & 0x0FFF
        elif (self.opcode & 0xF000) == 0x3000:
            if self.gpio[vx] == (self.opcode & 0x00FF):
                self.pc += 2
        elif (self.opcode & 0xF000) == 0x4000:
            if self.gpio[vx] != (self.opcode & 0x00FF):
                self.pc += 2
        elif (self.opcode & 0xF000) == 0x5000:
            if self.gpio[vx] == self.gpio[vy]:
                self.pc += 2
        elif (self.opcode & 0xF000) == 0x6000:
            self.gpio[vx] = self.opcode & 0x00FF
        elif (self.opcode & 0xF000) == 0x7000:
            self.gpio[vx] = (self.gpio[vx] + (self.opcode & 0x00FF)) & 0xFF
        elif (self.opcode & 0xF00F) == 0x8000:
            self.gpio[vx] = self.gpio[vy]
        elif (self.opcode & 0xF00F) == 0x8001:
            self.gpio[vx] |= self.gpio[vy]
        elif (self.opcode & 0xF00F) == 0x8002:
            self.gpio[vx] &= self.gpio[vy]
        elif (self.opcode & 0xF00F) == 0x8003:
            self.gpio[vx] ^= self.gpio[vy]
        elif (self.opcode & 0xF00F) == 0x8004:
            self.gpio[0xF] = 1 if self.gpio[vx] + self.gpio[vy] > 0xFF else 0
            self.gpio[vx] = (self.gpio[vx] + self.gpio[vy]) & 0xFF
        elif (self.opcode & 0xF00F) == 0x8005:
            self.gpio[0xF] = 0 if self.gpio[vy] > self.gpio[vx] else 1
            self.gpio[vx] = (self.gpio[vx] - self.gpio[vy]) & 0xFF
        elif (self.opcode & 0xF00F) == 0x8006:
            self.gpio[0xF] = self.gpio[vx] & 0x1
            self.gpio[vx] >>= 1
        elif (self.opcode & 0xF00F) == 0x8007:
            self.gpio[0xF] = 0 if self.gpio[vx] > self.gpio[vy] else 1
            self.gpio[vx] = (self.gpio[vy] - self.gpio[vx]) & 0xFF
        elif (self.opcode & 0xF00F) == 0x800E:
            self.gpio[0xF] = (self.gpio[vx] & 0x80) >> 7
            self.gpio[vx] = (self.gpio[vx] << 1) & 0xFF
        elif (self.opcode & 0xF00F) == 0x9000:
            if self.gpio[vx] != self.gpio[vy]:
                self.pc += 2
        elif (self.opcode & 0xF000) == 0xA000:
            self.index = self.opcode & 0x0FFF
        elif (self.opcode & 0xF000) == 0xB000:
            self.pc = (self.opcode & 0x0FFF) + self.gpio[0]
        elif (self.opcode & 0xF000) == 0xC000:
            self.gpio[vx] = random.randint(0, 255) & (self.opcode & 0x00FF)
        elif (self.opcode & 0xF000) == 0xD000:
            self._draw_sprite(vx, vy)
        elif (self.opcode & 0xF0FF) == 0xE09E:
            key = self.gpio[vx] & 0xF
            if self.key_inputs[key] == 1:
                self.pc += 2
                print(f"[DEBUG] Key check: VX={vx}, key={hex(key)} is PRESSED")
        elif (self.opcode & 0xF0FF) == 0xE0A1:
            key = self.gpio[vx] & 0xF
            if self.key_inputs[key] == 0:
                self.pc += 2
        elif (self.opcode & 0xF0FF) == 0xF007:
            self.gpio[vx] = self.delay_timer
        elif (self.opcode & 0xF0FF) == 0xF00A:
            key_pressed = False
            for i in range(16):
                if self.key_inputs[i] == 1:
                    self.gpio[vx] = i
                    key_pressed = True
                    break
            if not key_pressed:
                self.pc -= 2 
        elif (self.opcode & 0xF0FF) == 0xF015:
            self.delay_timer = self.gpio[vx]
        elif (self.opcode & 0xF0FF) == 0xF018:
            self.sound_timer = self.gpio[vx]
        elif (self.opcode & 0xF0FF) == 0xF01E:
            self.index = (self.index + self.gpio[vx]) & 0xFFF
        elif (self.opcode & 0xF0FF) == 0xF029:
            self.index = (self.gpio[vx] * 5) & 0xFFF
        elif (self.opcode & 0xF0FF) == 0xF033:
            self.memory[self.index] = self.gpio[vx] // 100
            self.memory[self.index + 1] = (self.gpio[vx] % 100) // 10
            self.memory[self.index + 2] = self.gpio[vx] % 10
        elif (self.opcode & 0xF0FF) == 0xF055:
            for i in range(vx + 1):
                self.memory[self.index + i] = self.gpio[i]
        elif (self.opcode & 0xF0FF) == 0xF065:
            for i in range(vx + 1):
                self.gpio[i] = self.memory[self.index + i]
        

        if self.delay_timer > 0:
            self.delay_timer -= 1
        if self.sound_timer > 0:
            self.sound_timer -= 1
        
        return True
    
    def _draw_sprite(self, vx, vy):
        self.gpio[0xF] = 0
        x = self.gpio[vx] & 0xFF
        y = self.gpio[vy] & 0xFF
        height = self.opcode & 0x000F
        
        for row in range(height):
            sprite_byte = self.memory[self.index + row]
            for col in range(8):
                if (y + row) >= 32 or (x + col) >= 64:
                    continue
                pixel = (sprite_byte >> (7 - col)) & 1
                loc = x + col + ((y + row) * 64)
                if pixel:
                    if self.display_buffer[loc] == 1:
                        self.gpio[0xF] = 1
                    self.display_buffer[loc] ^= 1
        self.should_draw = True


class CPUScheduler:
    def __init__(self, algorithm=SchedulingAlgorithm.ROUND_ROBIN, time_quantum=500):
        self.algorithm = algorithm
        self.time_quantum = time_quantum
        self.ready_queue = []
        self.lock = threading.Lock()
        self.total_processes = 0
        self.completed_processes = 0
        
    def add_process(self, pcb: ProcessControlBlock):
        with self.lock:
            pcb.state = ProcessState.READY
            self.ready_queue.append(pcb)
            self.total_processes += 1
            print(f"[SCHEDULER] Process {pcb.pid} ({pcb.name}) added to ready queue")
    
    def get_next_process(self) -> Optional[ProcessControlBlock]:
        with self.lock:
            if not self.ready_queue:
                return None
            
            if self.algorithm == SchedulingAlgorithm.FCFS:
                return self.ready_queue.pop(0)
            elif self.algorithm == SchedulingAlgorithm.PRIORITY:
                self.ready_queue.sort(key=lambda p: p.priority)
                return self.ready_queue.pop(0)
            else:
                return self.ready_queue.pop(0)
    
    def return_process(self, pcb: ProcessControlBlock):
        with self.lock:
            if pcb.state != ProcessState.TERMINATED:
                pcb.state = ProcessState.READY
                self.ready_queue.append(pcb)
    
    def mark_completed(self, pcb: ProcessControlBlock):
        with self.lock:
            pcb.state = ProcessState.TERMINATED
            self.completed_processes += 1
            print(f"[SCHEDULER] Process {pcb.pid} completed after {pcb.context_switches} context switches")


class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.next_pid = 1
        self.lock = threading.Lock()
    
    def create_process(self, name: str, rom_path: str, priority: int = 5) -> ProcessControlBlock:
        with self.lock:
            pid = self.next_pid
            self.next_pid += 1
            
            try:
                with open(rom_path, 'rb') as f:
                    program_data = f.read()
            except FileNotFoundError:
                print(f"ERROR: ROM file not found: {rom_path}")
                return None
            

            memory = [0] * 4096
            fonts = [
                0xF0, 0x90, 0x90, 0x90, 0xF0, 0x20, 0x60, 0x20, 0x20, 0x70,
                0xF0, 0x10, 0xF0, 0x80, 0xF0, 0xF0, 0x10, 0xF0, 0x10, 0xF0,
                0x90, 0x90, 0xF0, 0x10, 0x10, 0xF0, 0x80, 0xF0, 0x10, 0xF0,
                0xF0, 0x80, 0xF0, 0x90, 0xF0, 0xF0, 0x10, 0x20, 0x40, 0x40,
                0xF0, 0x90, 0xF0, 0x90, 0xF0, 0xF0, 0x90, 0xF0, 0x10, 0xF0,
                0xF0, 0x90, 0xF0, 0x90, 0x90, 0xE0, 0x90, 0xE0, 0x90, 0xE0,
                0xF0, 0x80, 0x80, 0x80, 0xF0, 0xE0, 0x90, 0x90, 0x90, 0xE0,
                0xF0, 0x80, 0xF0, 0x80, 0xF0, 0xF0, 0x80, 0xF0, 0x80, 0x80
            ]
            for i, byte in enumerate(fonts):
                memory[i] = byte
            for i, byte in enumerate(program_data):
                if i + 0x200 < 4096:
                    memory[i + 0x200] = byte
            
            pcb = ProcessControlBlock(
                pid=pid, name=name, state=ProcessState.NEW, priority=priority,
                program_counter=0x200, registers=[0] * 16, index_register=0,
                stack=[], memory=memory, display_buffer=[0] * 64 * 32,
                delay_timer=0, sound_timer=0, key_inputs=[0] * 16,
                burst_time=random.randint(100, 500), arrival_time=time.time()
            )
            
            self.processes[pid] = pcb
            print(f"[PROCESS MGR] Created process {pid} ({name}) from {rom_path}")
            return pcb
        

class DisplayWindow(pyglet.window.Window):
    def __init__(self, scheduler):
        super().__init__(640, 380, caption="CHIP-8 OS - Multi-Process Emulator")
        self.scheduler = scheduler
        self.current_display = [0] * 64 * 32
        self.current_process_name = "No Process"
        self.current_process = None
        self.current_cpu = None
        self.lock = threading.Lock()
        
        self.batch = pyglet.graphics.Batch()
        self.pixel_sprites = []
        
        pixel_data = b'\xff\xff\xff\xff' * (10 * 10)
        pixel_image = pyglet.image.ImageData(10, 10, 'RGBA', pixel_data)
        
        for i in range(2048):
            sprite = pyglet.sprite.Sprite(pixel_image, batch=self.batch)
            sprite.visible = False
            self.pixel_sprites.append(sprite)
        
        pyglet.clock.schedule_interval(self.update_display, 1/60.0)
    
    def update_process_display(self, display_buffer, process_name, process, cpu):
        with self.lock:
            self.current_display = display_buffer.copy()
            self.current_process_name = process_name
            self.current_process = process
            self.current_cpu = cpu
    
    def update_display(self, dt):
        pass
    
    def on_draw(self):
        self.clear()
        
        with self.lock:
            for y in range(32):
                for x in range(64):
                    i = x + (y * 64)
                    if i < len(self.current_display) and self.current_display[i] == 1:
                        pixel_x = x * 10
                        pixel_y = (31 - y) * 10
                        self.pixel_sprites[i].x = pixel_x
                        self.pixel_sprites[i].y = pixel_y
                        self.pixel_sprites[i].visible = True
                    else:
                        self.pixel_sprites[i].visible = False
        
        self.batch.draw()
        

        label = pyglet.text.Label(
            f'Process: {self.current_process_name} | Algorithm: {self.scheduler.algorithm.value}',
            font_name='Arial', font_size=12, x=10, y=340,
            color=(255, 255, 255, 255)
        )
        label.draw()
        
        stats = pyglet.text.Label(
            f'Completed: {self.scheduler.completed_processes}/{self.scheduler.total_processes} | Ready Queue: {len(self.scheduler.ready_queue)}',
            font_name='Arial', font_size=10, x=10, y=320,
            color=(200, 200, 200, 255)
        )
        stats.draw()
    
    def on_key_press(self, symbol, modifiers):
        with self.lock:
            if symbol in KEY_MAP:
                key_index = KEY_MAP[symbol]
                if self.current_process:
                    self.current_process.key_inputs[key_index] = 1
                if self.current_cpu:
                    self.current_cpu.key_inputs[key_index] = 1
                

                key_name = {
                    pyglet.window.key._1: "1", pyglet.window.key.Q: "Q", 
                    pyglet.window.key.W: "W", pyglet.window.key.E: "E",
                    pyglet.window.key.A: "A", pyglet.window.key.S: "S",
                    pyglet.window.key.D: "D", pyglet.window.key._4: "4",
                    pyglet.window.key.R: "R"
                }.get(symbol, f"Key-{hex(key_index)}")
                print(f"[INPUT] *** {key_name} PRESSED → CHIP-8 key {hex(key_index)} for {self.current_process_name} ***")
    
    def on_key_release(self, symbol, modifiers):
        with self.lock:
            if symbol in KEY_MAP:
                key_index = KEY_MAP[symbol]

                if self.current_process:
                    self.current_process.key_inputs[key_index] = 0
                if self.current_cpu:
                    self.current_cpu.key_inputs[key_index] = 0


class CPUExecutorThread(threading.Thread):
    def __init__(self, scheduler, cpu_id, display_window):
        super().__init__(daemon=True)
        self.scheduler = scheduler
        self.cpu_id = cpu_id
        self.cpu = CHIP8Core()
        self.display_window = display_window
        self.running = True
        
    def run(self):
        print(f"[CPU-{self.cpu_id}] Thread started")
        
        while self.running:
            pcb = self.scheduler.get_next_process()
            
            if pcb is None:
                time.sleep(0.01)
                continue
            
            # Context switch
            print(f"[CPU-{self.cpu_id}] Context switch to Process {pcb.pid} ({pcb.name})")
            pcb.state = ProcessState.RUNNING
            self.cpu.set_state(pcb.restore_context())
            
            self.display_window.update_process_display(pcb.display_buffer, pcb.name, pcb, self.cpu)
            
            start_time = time.time()
            instructions = 0
            
            for _ in range(self.scheduler.time_quantum):
                if not self.cpu.execute_instruction():
                    self.scheduler.mark_completed(pcb)
                    break
                instructions += 1
                
                if self.cpu.should_draw:
                    self.display_window.update_process_display(
                        self.cpu.display_buffer, pcb.name, pcb, self.cpu
                    )
                    self.cpu.should_draw = False
                
                time.sleep(0.002)
            else:
                pcb.save_context(self.cpu.get_state())
                pcb.cpu_time += time.time() - start_time
                self.scheduler.return_process(pcb)
    
    def stop(self):
        self.running = False

# Main OS
def main():
    time_quantum = 500
    rom_files = []
    
    for arg in sys.argv[1:]:
        if arg.startswith('--quantum='):
            time_quantum = int(arg.split('=')[1])
        else:
            rom_files.append(arg)
    
    print("\n" + "="*60)
    print("CHIP-8 OPERATING SYSTEM")
    print("Multi-Proces Emulator")
    print("="*60)
    print(f"ROM Files: {', '.join(rom_files)}")
    print(f"Scheduling: Round Robin (Time Quantum: {time_quantum} instructions = ~{time_quantum/500:.1f} seconds per game)")
    print(f"CPUs: 1 thread")
    print("="*60 + "\n")
    
    scheduler = CPUScheduler(SchedulingAlgorithm.ROUND_ROBIN, time_quantum=time_quantum)
    process_manager = ProcessManager()
    
    display = DisplayWindow(scheduler)
    
    for rom_file in rom_files:
        priority = random.randint(1, 5)
        pcb = process_manager.create_process(
            name=os.path.basename(rom_file),
            rom_path=rom_file,
            priority=priority
        )
        if pcb:
            scheduler.add_process(pcb)
        time.sleep(0.1)
    
    # Start CPU thread
    cpu_thread = CPUExecutorThread(scheduler, 0, display)
    cpu_thread.start()
    
    try:
        pyglet.app.run()
    except KeyboardInterrupt:
        pass
    
    cpu_thread.stop()
    cpu_thread.join(timeout=1)
    
    print("\n" + "="*60)
    print("SYSTEM STATISTICS")
    print("="*60)
    print(f"Total Processes: {scheduler.total_processes}")
    print(f"Completed: {scheduler.completed_processes}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
