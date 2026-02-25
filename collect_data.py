import glob
import sys
import time
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
import serial
import random

SAMPLING_RATE = 60.02  # Hz
RECORDING_DURATION = 3 # seconds
NUM_CHANNELS = 8

CYTON_BOARD_ID = 0
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'

NUM_TRIALS = 64

def find_openbci_port():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Error finding ports on your operating system')
    openbci_port = ''
    for port in ports:
        try:
            s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            time.sleep(2)
            if s.inWaiting():
                line = ''
                c = ''
                while '$$$' not in line:
                    c = s.read().decode('utf-8', errors='replace')
                    line += c
                if 'OpenBCI' in line:
                    openbci_port = port
            s.close()
        except (OSError, serial.SerialException):
            pass
    
    if openbci_port == '':
        raise OSError('Cannot find OpenBCI port.')
    
    return openbci_port

def record_eeg(serial_port):
    print("recording")
    
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = serial_port
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)

    board.start_stream(45000)
    time.sleep(RECORDING_DURATION)
    
    data = board.get_board_data()
    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    eeg = data[eeg_channels]
    
    board.stop_stream()
    board.release_session()
    
    print("recording done")
    return eeg

if __name__ == "__main__":
    port = find_openbci_port()
    recordings = []
    movements = ["left foot", "right foot", "tongue", "do nothing"]
    for i in range(NUM_TRIALS):
        trial_movement = movements[random.randint(0, len(movements) - 1)]
        print(f"trial {i + 1}/{NUM_TRIALS}")
        print(f"movement: {trial_movement}")
        eeg_data = record_eeg(port)
        recordings.append((trial_movement, eeg_data))
    
    outputarray = np.array(recordings, dtype=object)
    np.save("recordings.npy", outputarray)
