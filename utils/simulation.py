import pandas as pd
import time
from typing import Optional, Dict

class TimeTraveler:
    """
    Simulates real-time data streaming from a CSV file.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.current_step = 0
        self.max_steps = len(data)
        
    def next_tick(self) -> Optional[Dict]:
        """
        Returns the next data point (row) as a dictionary.
        Returns None if end of data.
        """
        if self.current_step >= self.max_steps:
            return None
            
        row = self.data.iloc[self.current_step].to_dict()
        # Ensure timestamp is string for JSON serialization if needed, or keep as obj
        # row['timestamp'] = str(row['timestamp']) 
        
        self.current_step += 1
        return row

    def reset(self):
        self.current_step = 0
        
    def get_progress(self) -> float:
        return self.current_step / self.max_steps

    def peek_future(self, steps: int) -> Optional[Dict]:
        """
        Look ahead 'steps' into the future without advancing.
        Returns the data point at current_step + steps.
        """
        target_step = self.current_step + steps
        if target_step >= self.max_steps:
            return None # End of data
            
        row = self.data.iloc[target_step].to_dict()
        return row
