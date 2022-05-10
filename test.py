import unittest
from cv2 import trace
import torch
from training.StatsTracker import StatsTracker

class TestStatsTracker(unittest.TestCase):
    # Make sure inititalization is correct
    def test_init(self):
        tracker = StatsTracker(16, 32)

        self.assertEqual(16, tracker.mean_denom_train)
        self.assertEqual(32, tracker.mean_denom_val)
    
    # Make sure val and train values are added properly
    def test_add_train_value(self):
        tracker = StatsTracker(32, 16)
        tracker.update_curr_losses(64, None)
        self.assertEqual(tracker.train_loss_curr, 64)
    def test_add_val_value(self):
        tracker = StatsTracker(32, 16)
        tracker.update_curr_losses(None, 14)
        self.assertEqual(tracker.val_loss_curr, 14)

    # Test multiple additions
    def test_add_multiple_train_values(self):
        tracker = StatsTracker(32, 16)
        for i in range(13):
            tracker.update_curr_losses(i, None) # train loss
        for j in range(21):
            tracker.update_curr_losses(None, j) # Val loss

        self.assertEqual(tracker.train_loss_curr, (12*(13))/2)

        self.assertEqual(tracker.val_loss_curr, (20*(21))/2)

    # Test mean computations
    def test_train_loss_mean(self):
        # TODO
        tracker = StatsTracker(39, 3)
        for i in range(13):
            tracker.update_curr_losses(i, None)
        for j in range(42):
            tracker.update_curr_losses(None, j)
        
        train_mean, val_mean = tracker.compute_means()

        self.assertEqual(train_mean, 2.0)

        self.assertEqual(val_mean, 287.0)

if __name__ == '__main__':
    unittest.main()