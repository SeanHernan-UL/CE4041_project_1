import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import the csv files form test run

training_df = pd.read_csv('mnist_test_20251103_232144/csv/mnist_training_20251103_232144.csv')
validation_df = pd.read_csv('mnist_test_20251103_232144/csv/mnist_validation_20251103_232144.csv')
test_df = pd.read_csv('mnist_test_20251103_232144/csv/mnist_test_20251103_232144.csv')

# bonus data from second run of 100
# TODO append this to the test_df
kanvar_test = [99.48999881744385, 99.36000108718872, 99.41999912261963, 99.4599997997284, 99.4599997997284, 99.34999942779541, 99.30999875068665, 99.39000010490417, 99.1599977016449, 99.26999807357788, 99.44999814033508, 99.41999912261963, 99.36000108718872, 99.25000071525574, 99.5199978351593, 99.44999814033508, 99.04000163078308, 99.27999973297119, 99.18000102043152, 99.33000206947327, 99.39000010490417, 99.44999814033508, 99.44000244140625, 99.47999715805054, 99.18000102043152, 99.40999746322632, 99.37000274658203, 99.26000237464905, 99.33000206947327, 99.40000176429749, 99.40999746322632, 99.41999912261963, 99.4599997997284, 99.33000206947327, 99.39000010490417, 99.44000244140625, 99.32000041007996, 99.47999715805054, 99.44000244140625, 99.47999715805054, 99.33000206947327, 99.40000176429749, 99.37999844551086, 99.34999942779541, 99.11999702453613, 99.51000213623047, 99.34999942779541, 99.37000274658203, 99.43000078201294, 99.44999814033508, 99.44999814033508, 99.3399977684021, 99.41999912261963, 99.48999881744385, 99.4599997997284, 99.36000108718872, 99.43000078201294, 99.22999739646912, 99.22999739646912, 99.32000041007996, 99.30999875068665, 99.43000078201294, 99.41999912261963, 99.44000244140625, 99.34999942779541, 99.44999814033508, 99.44000244140625, 99.2900013923645, 99.23999905586243, 99.44999814033508, 99.51000213623047, 99.40000176429749, 99.39000010490417, 99.21000003814697, 99.40000176429749, 99.26000237464905, 99.32000041007996, 99.08000230789185, 99.37000274658203, 99.37999844551086, 99.1100013256073, 99.43000078201294, 99.44999814033508, 99.6399998664856, 99.43000078201294, 99.44999814033508, 99.37999844551086, 99.43000078201294, 99.4599997997284, 99.36000108718872, 99.36000108718872, 99.33000206947327, 99.27999973297119, 99.44999814033508, 99.2900013923645, 99.41999912261963, 99.44999814033508, 99.4599997997284, 99.19999837875366, 99.22000169754028]
kanvar_test_df = pd.DataFrame(kanvar_test)

# preprocess training and validation
# get max value in each row... best epoch
training_df = training_df.max(axis='columns')
validation_df = validation_df.max(axis='columns')

## do boring stats stuff...
print('Assuming Gaussian distributions...')
print('Training Stats ------------------------------------------')
training_mean = float(training_df.mean())
training_min = float(training_df.min())
training_max = float(training_df.max())
training_std = float(training_df.std())

print('training_mean = {:6.3f}'.format(training_mean))
print('training_min = {:6.3f}'.format(training_min))
print('training_max = {:6.3f}'.format(training_max))
print('training_std = {:6.3f}'.format(training_std))

print('Test Stats ----------------------------------------------')
validation_mean = float(validation_df.mean())
validation_min = float(validation_df.min())
validation_max = float(validation_df.max())
validation_std = float(validation_df.std())

print('validation_mean = {:6.3f}'.format(validation_mean))
print('validation_min = {:6.3f}'.format(validation_min))
print('validation_max = {:6.3f}'.format(validation_max))
print('validation_std = {:6.3f}'.format(validation_std))

print('Test Stats ----------------------------------------------')
test_mean = float(test_df.mean())
test_min = float(test_df.min())
test_max = float(test_df.max())
test_std = float(test_df.std())

print('test_mean = {:6.3f}'.format(test_mean))
print('test_min = {:6.3f}'.format(test_min))
print('test_max = {:6.3f}'.format(test_max))
print('test_std = {:6.3f}'.format(test_std))
print('99.7 % of the data should fall within 3*std')
print('So accuracy should almost certainly be within [{:6.3f},'
      '{:6.3f}]'.format(test_mean-3*test_std,test_mean+3*test_std))

## Visualise the data
# create bar charts with 0.05 % step bins
bins = np.linspace(98.6, 99.6, 21)

plt.subplot(3,1,1)
plt.title('Histogram of MNIST test data (100 cycles)')
sns.histplot(training_df, bins=bins, color='#008080')
plt.legend(['training (max values)'])

plt.subplot(3,1,2)
sns.histplot(validation_df, bins=bins, color='#FF00FF')
plt.legend(['validation (max values)'])

plt.subplot(3,1,3)
sns.histplot(test_df, bins=bins, color='#FFD700')
plt.legend(['test'])

plt.xlabel('Accuracy (%)')

plt.show()