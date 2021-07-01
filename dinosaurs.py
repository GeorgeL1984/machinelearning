import numpy as np
from model_builder import ModelBuilder
from model_visualiser import ModelVisualiser

# Load CSV file containing dinosaur data. All values are loaded as strings.
data = np.genfromtxt('dinosaurs.csv', delimiter=',', dtype=None, encoding=None)

n_rows, n_cols = data.shape

print('data #columns:')
print(n_cols)

print('data #rows:')
print(n_rows)

# Label values to build model for (two because we are using perceptron
# with unit step activation function which can output two
# possible values).
label_values = ['Tyrannosaurus', 'Stegossaurus']

# Feature selection, column indexes of features excluding output  column.
all_feature_indexes = np.array(range(n_cols - 1))

n_dims = len(all_feature_indexes)
dim_1_offset = 1

# Build 2-dimensional model for each feature pair e.g [0,1], [0,2], etc
for dim_0 in range(0, n_dims - 1):
    for dim_1 in range(dim_1_offset, n_dims):
        feature_indexes = [dim_0, dim_1]
        model = ModelBuilder(data, feature_indexes, label_values)
        model.build()
        ModelVisualiser.visualise(model)

    dim_1_offset += 1

# Build n-dimensional model using all features.
model = ModelBuilder(data, all_feature_indexes, label_values)
model.build()
ModelVisualiser.visualise(model)

