import glob
import sys
import time
import numpy as np
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from serial import Serial
import serial
import random
import os

from threading import Thread, Event
from queue import Queue

from psychopy import visual, core
from psychopy.hardware import keyboard
from scipy import signal
import pickle

width = 1536
height = 864
aspect_ratio = width/height

import psychopy.visual
import psychopy.event
from psychopy import core

CYTON_SAMPLING_RATE = 250  # Hz
RECORDING_DURATION = 4.5   # seconds
BASELINE_DURATION = 1
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

SYNTHETIC_BOARD = True

win = psychopy.visual.Window(
    size=(width, height),
    units="norm",
    checkTiming = True,
    allowGUI = False,
    fullscr=False)

text = psychopy.visual.TextStim(win=win, height=0.145, color='white', units='norm', font="Helvetica")

def create_photosensor_dot(window, size=2/8*0.7):
    width, height = window.size
    ratio = width/height
    return visual.Rect(win=window, units="norm", width=size, height=size * ratio, 
                        fillColor='white', lineWidth = 0, pos = [1 - size/2, -1 - size/8]
    )
    
def init_window(win, dot):
    dot.color = np.array([-1, -1, -1])
    dot.draw()
    
    text.text = "init"
    text.draw()
    
    win.flip()

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
    print("Initializing Cyton")
    params = BrainFlowInputParams()
    if not SYNTHETIC_BOARD and CYTON_BOARD_ID != 6:
        params.serial_port = serial_port
    elif CYTON_BOARD_ID == 6:
        params.ip_port = 9000
    
    # board = BoardShim(CYTON_BOARD_ID, params)
    board_id = BoardIds.SYNTHETIC_BOARD.value
    board = BoardShim(board_id, params)

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

def get_data(board, queue_in, stop_event):
    while not stop_event.is_set():
        data_in = board.get_board_data()
        eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
        aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
        timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
        if len(timestamp_in) > 0:
            queue_in.put((eeg_in, aux_in, timestamp_in))
        time.sleep(0.1)

if __name__ == "__main__":
    photosensor_dot = create_photosensor_dot(win)
    init_window(win, photosensor_dot)
    
    if SYNTHETIC_BOARD:
        board = init_cyton()
    else:
        port = find_openbci_port()
        board = init_cyton(port)
    
    stop_event = Event()
    queue_in = Queue()

    eeg_continuous = np.zeros((NUM_CHANNELS, 0))
    aux_continuous = np.zeros((3, 0))
    trial_starts = np.array([], dtype=int)
    trial_ends = np.array([], dtype=int)
    
    baseline = []
    baseline_filtered = []
    recordings = []
    recordings_filtered = []
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # movements = ["stomp left", "stomp right", "flick tongue", "no action"]
    movements = ["stomp right", "no action"]
    try:
        # record a baseline
        print("preparing to record baseline...")
        time.sleep(PREP_DURATION)
        raw_data = record_bl(board)
        baseline.append(("no action", raw_data))
        
        board.start_stream(45000)
        cyton_thread = Thread(target=get_data, args=(board, queue_in, stop_event), daemon=True)
        cyton_thread.start()
        
        # run trials
        print("preparing to run trials...")
        time.sleep(PREP_DURATION)
        
        print("running trials")
        for i in range(NUM_TRIALS):
            trial_movement = movements[random.randint(0, len(movements) - 1)]

            # intertrial, draw label, black dot
            text.text = trial_movement
            text.draw()
            photosensor_dot.color = np.array([-1, -1, -1])
            photosensor_dot.draw()
            win.flip()
            time.sleep(0.75)

            print(f"trial {i}: {trial_movement}")
            # white dot
            text.text = trial_movement
            text.draw()
            photosensor_dot.color = np.array([1, 1, 1])
            photosensor_dot.draw()
            win.flip()
            time.sleep(RECORDING_DURATION)

            # black dot again
            photosensor_dot.color = np.array([-1, -1, -1])
            photosensor_dot.draw()
            win.flip()

            # process data
            if SYNTHETIC_BOARD:
                while not queue_in.empty():
                    print("reading fake data stream")
                    eeg_in, aux_in, timestamp_in = queue_in.get()
                    eeg_continuous = np.concatenate((eeg_continuous, eeg_in), axis=1)
                    aux_continuous = np.concatenate((aux_continuous, aux_in), axis=1)
                trial_samples = int(RECORDING_DURATION * CYTON_SAMPLING_RATE)
                raw_data = np.zeros((NUM_CHANNELS, trial_samples)) 
            else:
                while len(trial_ends) <= i:
                    print("reading data stream")
                    while not queue_in.empty():
                        eeg_in, aux_in, timestamp_in = queue_in.get()
                        eeg_continuous = np.concatenate((eeg_continuous, eeg_in), axis=1)
                        aux_continuous = np.concatenate((aux_continuous, aux_in), axis=1)
                    photo_trigger = (aux_continuous[1] > 20).astype(int)
                    trial_starts = np.where(np.diff(photo_trigger) == 1)[0]
                    trial_ends = np.where(np.diff(photo_trigger) == -1)[0]

                print("adding trial to recordings")
                trial_samples = int(RECORDING_DURATION * CYTON_SAMPLING_RATE)
                trial_start_sample = trial_starts[i]
                raw_data = np.copy(eeg_continuous[:, trial_start_sample:trial_start_sample + trial_samples])
                recordings.append((trial_movement, raw_data))
                print(f"Trial {i+1}/{NUM_TRIALS} ({trial_movement}): eeg shape {raw_data.shape}")
    finally:
        stop_event.set()
        board.stop_stream()
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

    np.save(f"{DATA_DIR}/eeg-continuous-session-{SES_NUMBER}.npy", eeg_continuous)
    np.save(f"{DATA_DIR}/aux-continuous-session-{SES_NUMBER}.npy", aux_continuous)