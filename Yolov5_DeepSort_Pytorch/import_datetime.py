from datetime import datetime, timedelta
from time import gmtime, strftime

now = datetime.now()
current_time = datetime.now()
print(datetime.now())
print("Current Time =", current_time)
print("Current Time next 10  =", current_time + timedelta(seconds=10))
print("Your Time Zone is GMT", strftime("%z", gmtime()))
if current_time <= current_time + timedelta(seconds=10):
    print('yes')