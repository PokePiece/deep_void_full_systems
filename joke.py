
import datetime

response = 'Would you like to hear a joke?'

print('Joke time')

user_input = input('Enter: ').strip()

if user_input == 'Which came first, the chicken or the egg':
    response = 'The egg.'
    
if user_input == 'Summarize general relativity':
    response = "Gravity isn't a force, it's geometry: mass warps space, and objects fall along curved trajectories which we call gravity."

if user_input == 'If you go to sleep at dawn, and wake up at noon, what happens soon?':
    response = 'A nap has occurred, so the evening awaits.'
    
currentdatetime = datetime.datetime.now()

print(currentdatetime)

print(response)