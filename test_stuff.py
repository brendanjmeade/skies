import datetime
import numpy as np
import h5py

# Matrix size and number of eigenvalues for truncated case
n_time_steps = 30
n_values = 1000
# fake_data = np.random.rand(n_time_steps, n_values)

# Open hdf file for writing
file_name = "test.hdf5"
hf = h5py.File(file_name, "w")

start_time = datetime.datetime.now()
ds = hf.create_dataset("geometric_moment", shape=(n_time_steps, n_values), dtype=float)
for i in range(n_time_steps):
    ds[i, :] = np.random.rand(n_values)
end_time = datetime.datetime.now()
print(f"HDF save time: {(end_time - start_time)}")



hf.close()
