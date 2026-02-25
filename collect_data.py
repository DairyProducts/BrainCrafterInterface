import glob
import sys
import time
import numpy as np
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from serial import Serial
import serial
import random
import os

CYTON_SAMPLING_RATE = 250  # Hz
RECORDING_DURATION = 3     # seconds
BASELINE_DURATION = 30
PREP_DURATION = 5
NUM_CHANNELS = 8

FILTER_LOW_FREQ = 8        # Hz
FILTER_HIGH_FREQ = 30      # Hz

CYTON_BOARD_ID = 0
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'

SES_NUMBER = 1
NUM_TRIALS = 64
DATA_DIR = "data/2-25"

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

def init_cyton(serial_port):
    print("Initializing Cyton")
    params = BrainFlowInputParams()
    if CYTON_BOARD_ID != 6:
        params.serial_port = serial_port
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    
    res = board.config_board('/0')
    print(res)
    res = board.config_board('//')
    print(res)
    res = board.config_board(ANALOGUE_MODE)
    print(res)

    print("Init done.")
    return board

def record_bl(board):
    print("Recording baseline:")
    board.start_stream(45000)
    time.sleep(BASELINE_DURATION)
    
    data = board.get_board_data()
    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    eeg = data[eeg_channels]
    
    board.stop_stream()
    
    print("Recording done.")
    return eeg

def record_eeg(board):
    print("Recording data:")
    board.start_stream(45000)
    time.sleep(RECORDING_DURATION)
    
    data = board.get_board_data()
    eeg_channels = board.get_eeg_channels(CYTON_BOARD_ID)
    eeg = data[eeg_channels]
    
    board.stop_stream()
    
    print("Recording done.")
    return eeg

def filter_eeg(raw_eeg):
    # Butterworth band pass filter
    filtered_eeg = mne.filter.filter_data(
        raw_eeg, 
        sfreq=CYTON_SAMPLING_RATE, 
        l_freq=FILTER_LOW_FREQ, 
        h_freq=FILTER_HIGH_FREQ, 
        method='iir',
        iir_params=dict(order=3, ftype='butter'),
        verbose=False
    )

    return filtered_eeg

if __name__ == "__main__":
    port = find_openbci_port()
    board = init_cyton(port)
    
    baseline = []
    baseline_filtered = []
    recordings = []
    recordings_filtered = []
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    movements = ["stomp left", "stomp right", "flick tongue", "no action"]
    try:
        # record a baseline
        print("preparing to record baseline...")
        time.sleep(PREP_DURATION)
        raw_data = record_bl(board)
        baseline.append(("no action", raw_data))
        
        # run trials
        print("preparing to run trials...")
        time.sleep(PREP_DURATION)
        
        print("running trials")
        for i in range(NUM_TRIALS):
            trial_movement = movements[random.randint(0, len(movements) - 1)]
            print("-" * 20)
            print(f"trial {i + 1}/{NUM_TRIALS}")
            print(f"movement: {trial_movement}")
            # TODO: delay here? discuss in lab
            raw_data = record_eeg(board)
            recordings.append((trial_movement, raw_data))
    finally:
        board.release_session()
    
    # filter data. note recordings of form [(tm, data), ...]
    for i in range(len(recordings)):
        filtered_data = filter_eeg(recordings[i][1])
        recordings_filtered.append((recordings[i][0], filtered_data))
        
    filtered_data = filter_eeg(baseline[0][1])
    baseline_filtered.append((baseline[0][0], filtered_data))

    
    # save to files
    base_output = np.array(baseline, dtype=object)
    np.save(f"{DATA_DIR}/baseline-session-{SES_NUMBER}.npy", base_output)
    filtered_base_output = np.array(baseline_filtered, dtype=object)
    np.save(f"{DATA_DIR}/filtered-baseline-session-{SES_NUMBER}.npy", filtered_base_output)
    raw_output = np.array(recordings, dtype=object)
    np.save(f"{DATA_DIR}/raw-session-{SES_NUMBER}.npy", raw_output)
    filtered_output = np.array(recordings_filtered, dtype=object)
    np.save(f"{DATA_DIR}/filtered-session-{SES_NUMBER}.npy", filtered_output)
