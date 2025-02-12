from __future__ import print_function, division

"""
Send trigger events to parallel port.
See sample code at the end.
Kyuhwa Lee, 2014
Swiss Federal Institute of Technology Lausanne (EPFL)

Modified for Python 3.10 so that it no longer depends on the old 'parallel'
library. In the USB2LPT (Linux) branch we now use Python’s multiprocessing to
open and write to "/dev/parport0".
"""

import threading
import platform
import os
import sys
import ctypes
import time
from builtins import input, bytes

# --- New imports for the Linux multiprocessing–based port driver ---
import multiprocessing
from queue import Empty


# This class implements a very minimal parallel port controller for Linux.
# It runs as a separate process and waits for trigger values on a Queue.
class LinuxParallelPortProcess(multiprocessing.Process):
    def __init__(self, portaddr, queue):
        super().__init__()
        self.portaddr = portaddr
        self.queue = queue
        self._stop_event = multiprocessing.Event()

    def run(self):
        try:
            # Open the parallel port device file in binary write mode.
            # (You may need to adjust buffering or permissions for your system.)
            with open(self.portaddr, "wb", buffering=0) as f:
                while not self._stop_event.is_set():
                    try:
                        # Wait for a value (with timeout so we can check _stop_event)
                        value = self.queue.get(timeout=0.1)
                        f.write(bytes([value]))
                        f.flush()
                    except Empty:
                        continue
        except Exception as e:
            print("Error in LinuxParallelPortProcess:", e)

    def stop(self):
        self._stop_event.set()


class Trigger(object):
    """
    Supported device types:
     'Arduino': CNBI Arduino trigger
     'USB2LPT': Commercial USB2LPT adapter
     'DESKTOP': Desktop native LPT
     'SOFTWARE': Software trigger
     'FAKE': Mock trigger device for testing
    ...
    """
    def __init__(self, lpttype='USB2LPT', portaddr=None, verbose=False):
        self.evefile = None
        self.lpttype = lpttype
        self.verbose = verbose
        self.offprocess = None  # used for Linux USB2LPT delay

        if self.lpttype in ['USB2LPT', 'DESKTOP'] and platform.system() == "Windows":
            if self.lpttype == 'USB2LPT':
                if ctypes.sizeof(ctypes.c_voidp) == 4:
                    dllname = 'LptControl_USB2LPT32.dll'  # 32 bit
                else:
                    dllname = 'LptControl_USB2LPT64.dll'  # 64 bit
                if portaddr not in [0x278, 0x378]:
                    self.print('Warning: LPT port address %d is unusual.' % portaddr)

            elif self.lpttype == 'DESKTOP':
                if ctypes.sizeof(ctypes.c_voidp) == 4:
                    dllname = 'LptControl_Desktop32.dll'  # 32 bit
                else:
                    dllname = 'LptControl_Desktop64.dll'  # 64 bit
                if portaddr not in [0x278, 0x378]:
                    self.print('Warning: LPT port address %d is unusual.' % portaddr)

            self.portaddr = portaddr
            search = []
            search.append(os.path.join(os.path.dirname(__file__), dllname))
            search.append(os.path.join(os.path.dirname(__file__), 'libs', dllname))
            search.append(os.path.join(os.getcwd(), dllname))
            search.append(os.path.join(os.getcwd(), 'libs', dllname))
            for f in search:
                if os.path.exists(f):
                    dllpath = f
                    break
            else:
                self.print('ERROR: Cannot find the required library %s' % dllname)
                raise RuntimeError

            self.print('Loading %s' % dllpath)
            self.lpt = ctypes.cdll.LoadLibrary(dllpath)

        # --- New Linux branch: use multiprocessing instead of the 'parallel' library ---
        elif self.lpttype == "USB2LPT" and platform.system() == "Linux":
            self.portaddr = "/dev/parport0"
            # Create a Queue for sending trigger values to the port process.
            self._pp_queue = multiprocessing.Queue()
            self.lpt = LinuxParallelPortProcess(self.portaddr, self._pp_queue)
            self.lpt.start()

        elif self.lpttype == 'ARDUINO':
            import serial, serial.tools.list_ports
            BAUD_RATE = 115200

            # portaddr should be None or in the form of 'COM1', 'COM2', etc.
            if portaddr is None:
                arduinos = [x for x in serial.tools.list_ports.grep('Arduino')]
                if len(arduinos) == 0:
                    print('No Arduino found. Stop.')
                    sys.exit()

                for i, a in enumerate(arduinos):
                    print('Found', a[0])
                try:
                    com_port = arduinos[0].device
                except AttributeError:  # depends on Python distribution
                    com_port = arduinos[0][0]
            else:
                com_port = portaddr

            self.ser = serial.Serial(com_port, BAUD_RATE)
            time.sleep(1)  # sometimes necessary for stabilization
            print('Connected to %s.' % com_port)

        elif self.lpttype == 'SOFTWARE':
            from pycnbi.stream_receiver.stream_receiver import StreamReceiver
            self.print('Using software trigger')

            # get data file location
            LSL_SERVER = 'StreamRecorderInfo'
            inlet = cnbi_lsl.start_client(LSL_SERVER)
            fname = inlet.info().source_id()
            if fname[-4:] != '.pcl':
                self.print('ERROR: Received wrong record file name format %s' % fname)
                sys.exit(-1)
            evefile = fname[:-8] + '-eve.txt'
            eveoffset_file = fname[:-8] + '-eve-offset.txt'
            self.print('Event file is: %s' % evefile)
            self.evefile = open(evefile, 'a')

            # check server LSL time server integrity
            self.print("Checking LSL server's timestamp integrity for logging software triggers.")
            amp_name, amp_serial = pu.search_lsl()
            sr = StreamReceiver(window_size=1, buffer_size=1, amp_serial=amp_serial, eeg_only=False, amp_name=amp_name)
            local_time = pylsl.local_clock()
            server_time = sr.get_window_list()[1][-1]
            lsl_time_offset = local_time - server_time
            with open(eveoffset_file, 'a') as f:
                f.write('Local time: %.6f, Server time: %.6f, Offset: %.6f\n' % (local_time, server_time, lsl_time_offset))
            self.print('LSL timestamp offset (%.3f) saved to %s' % (lsl_time_offset, eveoffset_file))

        elif self.lpttype == 'FAKE' or self.lpttype is None or self.lpttype is False:
            self.print('WARNING: Using a fake trigger.')
            self.lpttype = 'FAKE'
            self.lpt = None

        else:
            self.print('ERROR: Unknown LPT port type %s' % lpttype)
            sys.exit(-1)

    def __del__(self):
        if self.evefile is not None and not self.evefile.closed:
            self.evefile.close()
            self.print('Event file saved.')
            sys.stdout.flush()
        # For Linux USB2LPT: stop the background port process.
        if self.lpttype == 'USB2LPT' and platform.system() == "Linux":
            if self.lpt is not None:
                self.lpt.stop()
                self.lpt.join()

    def print(self, *args):
        # Replace qc.print_c with a plain print if you don’t have qc installed.
        print('[pyLptControl] ', end='')
        print(*args)

    def init(self, duration):
        if self.lpttype == 'SOFTWARE':
            self.print('>> Ignoring delay parameter for software trigger')
            return True
        elif self.lpttype == 'FAKE':
            return True
        else:
            self.delay = duration / 1000.0  # delay in seconds

            if self.lpttype in ['DESKTOP', 'USB2LPT'] and platform.system() == "Windows":
                if self.lpt.init() == -1:
                    self.print('Connecting to LPT port failed. Check the driver status.')
                    self.lpt = None
                    return False
                # Use a threading.Timer for Windows.
                self.offtimer = threading.Timer(self.delay, self.signal_off)
            elif self.lpttype == 'USB2LPT' and platform.system() == "Linux":
                # For Linux, we will launch a short-lived process to send 0 after the delay.
                self.offprocess = None
            return True

    # write to software trigger
    def write_event(self, value):
        assert self.lpttype == 'SOFTWARE'
        # (Assuming pylsl.local_clock() is defined in your environment)
        self.evefile.write('%.6f\t0\t%d\n' % (pylsl.local_clock(), value))
        return True

    # set data
    def set_data(self, value):
        if self.lpttype == 'SOFTWARE':
            self.print('>> set_data() not supported for software trigger.')
            return False
        elif self.lpttype == 'FAKE':
            self.print('FAKE trigger value', value)
            return True
        else:
            if self.lpttype == 'USB2LPT' and platform.system() == "Windows":
                self.lpt.setdata(value)
            # --- In the Linux branch we simply put the value into our Queue ---
            elif self.lpttype == 'USB2LPT' and platform.system() == "Linux":
                self._pp_queue.put(value)
            elif self.lpttype == 'DESKTOP':
                self.lpt.setdata(self.portaddr, value)
            elif self.lpttype == 'ARDUINO':
                self.ser.write(bytes([value]))
            else:
                raise RuntimeError('Wrong trigger device')

    # sends data and turns off after delay
    def signal(self, value):
        if self.lpttype == 'SOFTWARE':
            if self.verbose is True:
                self.print('Sending software trigger', value)
            return self.write_event(value)
        elif self.lpttype == 'FAKE':
            self.print('Sending FAKE trigger signal', value)
            return True
        else:
            # --- For Linux USB2LPT, use a short-lived process for the off signal ---
            if self.lpttype == 'USB2LPT' and platform.system() == "Linux":
                if self.offprocess is not None and self.offprocess.is_alive():
                    print('Warning: You are sending a new signal before the last one finished. Signal ignored.')
                    print('self.delay=%.1f' % self.delay)
                    return False
                self.set_data(value)
                if self.verbose is True:
                    self.print('Sending', value)
                # Start a new process to wait the delay and then call signal_off
                self.offprocess = multiprocessing.Process(target=self._delayed_signal_off)
                self.offprocess.start()
                return True
            else:
                # --- Windows (and other) branch using threading.Timer ---
                if self.offtimer.is_alive():
                    print('Warning: You are sending a new signal before the last one finished. Signal ignored.')
                    print('self.delay=%.1f' % self.delay)
                    return False
                self.set_data(value)
                if self.verbose is True:
                    self.print('Sending', value)
                self.offtimer.start()
                return True

    def _delayed_signal_off(self):
        time.sleep(self.delay)
        self.signal_off()

    # set data to zero (all bits off)
    def signal_off(self):
        if self.lpttype == 'SOFTWARE':
            return self.write_event(0)
        elif self.lpttype == 'FAKE':
            self.print('FAKE trigger off')
            return True
        else:
            self.set_data(0)
            # For Windows, reinitialize the timer
            if platform.system() == "Windows":
                self.offtimer = threading.Timer(self.delay, self.signal_off)

    # set pin
    def set_pin(self, pin):
        if self.lpttype == 'SOFTWARE':
            self.print('>> set_pin() not supported for software trigger.')
            return False
        elif self.lpttype == 'FAKE':
            self.print('FAKE trigger pin', pin)
            return True
        else:
            self.set_data(2 ** (pin - 1))


class MockTrigger(object):
    def __init__(self):
        self.print('*' * 50)
        self.print(' WARNING: MockTrigger class is deprecated.')
        self.print("          Use Trigger('FAKE') instead.")
        self.print('*' * 50 + '\n')

    def init(self, duration=100):
        self.print('Mock Trigger ready')
        return True

    def print(self, *args):
        print('[pyLptControl] ', end='')
        print(*args)

    def signal(self, value):
        self.print('FAKE trigger signal', value)
        return True

    def signal_off(self):
        self.print('FAKE trigger value 0')
        return True

    def set_data(self, value):
        self.print('FAKE trigger value', value)
        return True

    def set_pin(self, pin):
        self.print('FAKE trigger pin', pin)
        return True


# set 1 to each bit and rotate from bit 0 to bit 7
def test_all_bits(trigger):
    for x in range(8):
        val = 2 ** x
        trigger.signal(val)
        print(val)
        time.sleep(1)


# sample test code
if __name__ == '__main__':
    # Example: for Arduino trigger use: trigger = Trigger('ARDUINO', 'COM3')
    # For USB2LPT trigger (Windows or Linux) use:
    trigger = Trigger('USB2LPT')
    if not trigger.init(200):
        print('LPT port cannot be opened. Using mock trigger.')
        trigger = MockTrigger()

    # Uncomment to test bit-by-bit triggering:
    # test_all_bits(trigger)

    print('Type quit or Ctrl+C to finish.')
    while True:
        val = input('Trigger value? ')
        if val.strip() == '':
            continue
        if val == 'quit':
            break
        try:
            intval = int(val)
        except ValueError:
            print('Ignored %s' % val)
            continue
        if 0 <= intval <= 255:
            trigger.signal(intval)
            print('Sent %d' % intval)
        else:
            print('Ignored %s' % val)
