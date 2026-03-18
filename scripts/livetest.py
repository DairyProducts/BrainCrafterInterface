import sys
import time
import glob
from threading import Thread, Event

import serial
from serial import Serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import numpy as np
import mne
from mne.decoding import CSP
import joblib
from pynput.keyboard import Controller as KbController

FAKE_BOARD = False

CYTON_SAMPLING_RATE = 250
NUM_CHANNELS = 8
FILTER_LOW_FREQ = 8
FILTER_HIGH_FREQ = 30
CYTON_BOARD_ID = 0
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'

MODEL_PATH = "models/3-4joshstomp.pkl"
CHANNELS = [2, 3, 4, 6]
WINDOW_SEC = 4
WINDOW_SAMPLES = WINDOW_SEC * CYTON_SAMPLING_RATE
BUFFER_SEC = 12
BUFFER_SAMPLES = BUFFER_SEC * CYTON_SAMPLING_RATE

# openbci/Cyton interfacing helpers
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

def init_cyton(serial_port=None):
    print("Initializing Cyton...")
    params = BrainFlowInputParams()
    
    if FAKE_BOARD or CYTON_BOARD_ID == 6:
        print("Board is synthetic.")
        params.ip_port = 9000
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        print("Board is real.")
        params.serial_port = serial_port
        board_id = CYTON_BOARD_ID

    board = BoardShim(board_id, params)

    board.prepare_session()
    print("Configuring Cyton...")
    
    res = board.config_board('/0')
    print(res)
    res = board.config_board('//')
    print(res)
    res = board.config_board(ANALOGUE_MODE)
    print(res)

    print("Cyton setup done.")
    return board

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


def stream_data(board, eeg, stop_event):
    board_id = board.board_id
    while not stop_event.is_set():
        data = board.get_board_data()
        new_data = data[board.get_eeg_channels(board_id)]
        if new_data.shape[1] > 0:
            combined = np.concatenate((eeg[0], new_data), axis=1)
            eeg[0] = combined[:, -BUFFER_SAMPLES:] if combined.shape[1] > BUFFER_SAMPLES else combined
        time.sleep(0.1)

if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    print(f"model loaded from {MODEL_PATH}")

    port = find_openbci_port()
    board = init_cyton(port)

    eeg = [np.zeros((NUM_CHANNELS, 0))]

    stop_event = Event()

    board.start_stream(45000)
    stream_thread = Thread(
        target=stream_data,
        args=(board, eeg, stop_event),
        daemon=True,
    )
    stream_thread.start()
    keyboard = KbController()
    print("ready and started")

    try:
        while True:
            # "yes i basically wrote a race condition here" -derek
            current = eeg[0].copy()
            if current.shape[1] < WINDOW_SAMPLES:
                time.sleep(0.5)
                continue

            window = current[:, -WINDOW_SAMPLES:]
            filtered = filter_eeg(window)
            segment = filtered[CHANNELS, 62:-100]
            
            prediction = model.predict(segment[np.newaxis, :, :])[0]

            if prediction == 1:
                print("omg it works uwaaaaah????")
                keyboard.press('e')
                time.sleep(0.1) # immediate press and release may nto work in some games
                keyboard.release('e')
                time.sleep(WINDOW_SEC)
            else:
                time.sleep(0.1)
    finally:
        stop_event.set()
        board.stop_stream()
        board.release_session()
