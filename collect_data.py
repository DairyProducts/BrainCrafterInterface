import os
import sys
import time
import random
import glob
from threading import Thread, Event
from queue import Queue

import serial
from serial import Serial
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import psychopy.visual

import numpy as np
import mne

# things to change
WIDTH = 1536               # px
HEIGHT = 864               # px
RECORDING_DURATION = 4.5   # seconds
BASELINE_DURATION = 5      # seconds
PREP_DURATION = 5          # seconds
BREAK_DURATION = 0.75      # seconds
SES_NUMBER = 2
NUM_TRIALS = 5
DATA_DIR = "data/2-26"
FAKE_BOARD = True

# things not NOT change
CYTON_SAMPLING_RATE = 250  # Hz
NUM_CHANNELS = 8
FILTER_LOW_FREQ = 8        # Hz
FILTER_HIGH_FREQ = 30      # Hz
CYTON_BOARD_ID = 0
BAUD_RATE = 115200
ANALOGUE_MODE = '/2'

win = psychopy.visual.Window(
        size=(WIDTH, HEIGHT),
        units="norm",
        checkTiming = True,
        allowGUI = False,
        fullscr=False
    )

text = psychopy.visual.TextStim(win=win, height=0.145, color='white', units='norm', font="Helvetica")

# psychopy helpers (i hate psychopy bruh)
def init_photosensor_dot(window, size=2/8*0.7):
    w, h = window.size
    ratio = w/h
    return psychopy.visual.Rect(win=window, units="norm", width=size, height=size * ratio, 
                        fillColor='white', lineWidth = 0, pos = [1 - size/2, -1 - size/8]
    )

photosensor_dot = init_photosensor_dot(win)
    
def init_window():
    # draw dot
    photosensor_dot.color = np.array([-1, -1, -1])
    photosensor_dot.draw()
    # draw some text
    text.text = "init"
    text.draw()
    # ~~draw~~ flip window
    win.flip()

def change_window(color_arr, s=None):
    photosensor_dot.color = np.array(color_arr)
    photosensor_dot.draw()
    
    if s is not None:
        text.text = s
        text.draw()
    
    win.flip()

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
        board = BoardShim(board_id, params)
    else:
        print("Board is real.")
        params.serial_port = serial_port
        board = BoardShim(CYTON_BOARD_ID, params)

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

def get_data(board, queue_in, stop_event):
    while not stop_event.is_set():
        data_in = board.get_board_data()
        eeg_in = data_in[board.get_eeg_channels(CYTON_BOARD_ID)]
        aux_in = data_in[board.get_analog_channels(CYTON_BOARD_ID)]
        timestamp_in = data_in[board.get_timestamp_channel(CYTON_BOARD_ID)]
        if len(timestamp_in) > 0:
            queue_in.put((eeg_in, aux_in, timestamp_in))
        time.sleep(0.1)

# data processing helpers
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

# main script
if __name__ == "__main__":
    init_window()
    
    if FAKE_BOARD:
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
    
    black = [-1, -1, -1]
    white = [1, 1, 1]
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # movements = ["stomp left", "stomp right", "flick tongue", "no action"]
    movements = ["stomp right", "no action"]
    try:
        # record a baseline
        change_window(black, f"collecting baseline in {str(PREP_DURATION)}")
        print("Preparing to record baseline...")
        time.sleep(PREP_DURATION)
        
        change_window(white, "baseline")
        
        raw_data = record_bl(board)
        baseline.append(("no action", raw_data))
        
        change_window(black)
        
        board.start_stream(45000)
        cyton_thread = Thread(target=get_data, args=(board, queue_in, stop_event), daemon=True)
        cyton_thread.start()
        
        # run trials
        print("Preparing to run trials...")
        change_window(black, f"running trials in {str(PREP_DURATION)}")
        time.sleep(PREP_DURATION)
        
        for i in range(NUM_TRIALS):
            trial_movement = movements[random.randint(0, len(movements) - 1)]
            
            # intertrial: black dot, draw movement to screen
            change_window(black, trial_movement)
            time.sleep(BREAK_DURATION)
            # trial: white dot
            print(f"Trial {i+1}/{NUM_TRIALS}: {trial_movement}")
            change_window(white, trial_movement)
            time.sleep(RECORDING_DURATION)
            # posttrial: black dot, remove movement to prep for next one
            change_window(black)
            
            # data collection
            if FAKE_BOARD:
                while not queue_in.empty():
                    eeg_in, aux_in, timestamp_in = queue_in.get()
                    eeg_continuous = np.concatenate((eeg_continuous, eeg_in), axis=1)
                    aux_continuous = np.concatenate((aux_continuous, aux_in), axis=1)
                
                print("Adding trial to recordings...")
                trial_samples = int(RECORDING_DURATION * CYTON_SAMPLING_RATE)
                raw_data = np.zeros((NUM_CHANNELS, trial_samples)) 
                recordings.append((trial_movement, raw_data))
                print(f"Trial {i+1}/{NUM_TRIALS} ({trial_movement}): eeg shape {raw_data.shape} saved.")
            else:
                while len(trial_ends) <= i:
                    while not queue_in.empty():
                        eeg_in, aux_in, timestamp_in = queue_in.get()
                        eeg_continuous = np.concatenate((eeg_continuous, eeg_in), axis=1)
                        aux_continuous = np.concatenate((aux_continuous, aux_in), axis=1)
                    # this print for debugging
                    print(f"aux ch1 range: min={aux_continuous[1].min():.2f} max={aux_continuous[1].max():.2f}")
                    photo_trigger = (aux_continuous[1] > 20).astype(int)
                    trial_starts = np.where(np.diff(photo_trigger) >= 0.9)[0]
                    trial_ends = np.where(np.diff(photo_trigger) <= -0.9)[0]

                print("Adding trial to recordings...")
                trial_samples = int(RECORDING_DURATION * CYTON_SAMPLING_RATE)
                trial_start_sample = trial_starts[i]
                raw_data = np.copy(eeg_continuous[:, trial_start_sample:trial_start_sample + trial_samples])
                recordings.append((trial_movement, raw_data))
                print(f"Trial {i+1}/{NUM_TRIALS} ({trial_movement}): eeg shape {raw_data.shape} saved.")
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
    print(f"{DATA_DIR}/baseline-session-{SES_NUMBER} saved.")
    
    filtered_base_output = np.array(baseline_filtered, dtype=object)
    np.save(f"{DATA_DIR}/filtered-baseline-session-{SES_NUMBER}.npy", filtered_base_output)
    print(f"{DATA_DIR}/filtered-baseline-session-{SES_NUMBER} saved.")
    
    raw_output = np.array(recordings, dtype=object)
    np.save(f"{DATA_DIR}/raw-session-{SES_NUMBER}.npy", raw_output)
    print(f"{DATA_DIR}/raw-session-{SES_NUMBER} saved.")
    
    filtered_output = np.array(recordings_filtered, dtype=object)
    np.save(f"{DATA_DIR}/filtered-session-{SES_NUMBER}.npy", filtered_output)
    print(f"{DATA_DIR}/filtered-session-{SES_NUMBER} saved.")

    np.save(f"{DATA_DIR}/eeg-continuous-session-{SES_NUMBER}.npy", eeg_continuous)
    np.save(f"{DATA_DIR}/aux-continuous-session-{SES_NUMBER}.npy", aux_continuous)
    print("All done!")
