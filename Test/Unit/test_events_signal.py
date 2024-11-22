import unittest, argparse
import numpy as np   
from pymgipsim.InputGeneration.signal import Signal, Events

class TestEvents(unittest.TestCase):

    def setUp(self):
        pass

    def testInconsistentDimensions(self):
        with self.assertRaises(AssertionError):
            Events(duration=np.ones((3,5)), start_time=np.ones((3,4)), magnitude=np.ones((3,4)))
        with self.assertRaises(AssertionError):
            Events(start_time=np.ones((1,4)), magnitude=np.ones((3,4)))

    def testSigns(self):
        with self.assertRaises(AssertionError):
            Events(duration=-1*np.ones((3,5)), start_time=np.ones((3,5)), magnitude=np.ones((3,5)))
        with self.assertRaises(AssertionError):
            Events(duration=np.ones((3,5)), start_time=-1*np.ones((3,5)), magnitude=np.ones((3,5)))
        with self.assertRaises(AssertionError):
            Events(duration=np.ones((3,5)), start_time=np.ones((3,5)), magnitude=-1*np.ones((3,5)))



class TestSignal(unittest.TestCase):

    def testMeasurement(self):
        values = np.expand_dims(np.linspace(0, 1000, 21), 0)
        start_times = np.expand_dims(
            np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 60, 70]), 0)
        sampling_time = 0.5
        time = np.linspace(0, 80, 161)
        signal = Signal(time=time, magnitude=values, start_time=start_times, sampling_time=sampling_time)
        ref = np.load('Test/Unit/Files/signal_test1.npy')
        assert np.array_equal(signal.sampled_signal, ref)

    def testDuration(self):
        values = np.expand_dims(np.asarray([3, 6, 9]), 0)
        start_times = np.expand_dims(np.asarray([0, 20, 21]), 0)
        durations = np.expand_dims(np.asarray([1.5, 3, 3]), 0)
        sampling_time = 0.5
        time = np.linspace(0, 80, 161)
        signal = Signal(time=time, magnitude=values, start_time=start_times, sampling_time=sampling_time,
                        duration=durations)
        ref = np.load("Test/Unit/Files/signal_test2.npy")
        assert np.array_equal(signal.sampled_signal, ref)


if __name__ == '__main__':
    unittest.main()