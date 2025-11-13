import serial
import time
import serial.tools.list_ports
import platform

#Variables
pos = 0
#Arduino Port detection
def find_arduino():
    """Find Arduino automatically"""
    for port in serial.tools.list_ports.comports():
        if 'Arduino' in port.description or 'CH340' in port.description or 'ttyUSB' in port.description:
            return port.device
    return None



if platform.system() == "Windows":
    port = find_arduino()        # typical on Windows
else:
    port = "/dev/ttyUSB0"  # typical on Linux


if port:
    print(f"Found Arduino on {port}")
else:
    print("Arduino not found")


ser = serial.Serial(port, baudrate = 9600, timeout=1)
time.sleep(2)

#check if arduino sent something
def recieve_Message():
        if ser.in_waiting > 0:
                        arduino_message = ser.readline().decode('utf-8', errors='ignore').strip()
                        if arduino_message:
                                print(f"Arduino: {arduino_message}")
                                time.sleep(0.1)
#get user input and send Message
def send_Message(message):
    ser.write((str(message) + '\n').encode('utf-8'))
    time.sleep(0.1)
    return True

#Andreis methods
def goback(x=1):
    global pos
    target = pos - (x * 20)  # move back by x * 20 degrees
    if target < 0:
        target = 0  # limit to minimum position
    for angle in range(pos, target - 1, -20):  # smooth steps backward
        pos = angle
        send_Message(pos)
        time.sleep(0.05)
def goforward(x=1):
    global pos
    target = pos + (x * 20)  # move forward by 20 degrees
    if target > 180:
        target = 180  # limit to max position
    for angle in range(pos, target + 1, 20):  # smooth steps of 2 degrees
        pos = angle
        send_Message(pos)
        time.sleep(0.05)  # adjust delay for speed
def turnZero():
    global pos
    for angle in range(pos, -1, -10):  # go from current pos down to 0 in steps of -10
        pos = angle
        send_Message(pos)
        time.sleep(0.1)
def turnMaximum():
    global pos
    for angle in range(pos, 181, 10):  # go from current pos to 180 in steps of 10
        pos = angle
        send_Message(pos)
        time.sleep(0.1) 
        
try:
    goforward()
    time.sleep(1)
    goforward()
    time.sleep(1)
    goback()
    time.sleep(1)
    goback()

except KeyboardInterrupt:
        print("\nProgramm stopped by user.")
finally:
        ser.close()
        print("Serial closed.")