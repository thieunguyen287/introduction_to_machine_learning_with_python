import pandas as pd
import numpy as np

data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
        'Location': ['New York', 'Paris', 'Berlin', 'London'],
        'Age': [24, 13, 53, 33]}

data_frame = pd.DataFrame(data)
print data_frame

print data_frame[data_frame.Age > 30]

