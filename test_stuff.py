import datetime
import numpy as np
import h5py



# # Matrix size and number of eigenvalues for truncated case
# n_time_steps = 30
# n_values = 1000
# # fake_data = np.random.rand(n_time_steps, n_values)

# # Open hdf file for writing
# file_name = "test.hdf5"
# hf = h5py.File(file_name, "w")

# start_time = datetime.datetime.now()
# ds = hf.create_dataset("geometric_moment", shape=(n_time_steps, n_values), dtype=float)
# for i in range(n_time_steps):
#     ds[i, :] = np.random.rand(n_values)
# end_time = datetime.datetime.now()
# print(f"HDF save time: {(end_time - start_time)}")



def moment_to_moment_magnitude(moment):
    # moment_magnitude = 2/3 * (np.log10(moment + 7.0) - 10.7)
    moment_magnitude = 2/3 * (np.log10(moment) - 16.1)
    # moment_magnitude = 2/3 * (np.log10(moment) - 16.1)    
    return moment_magnitude


# 10000000000
length = 100 * 1e3 * 1e2
depth = 10 * 1e3  * 1e2
shear_modulus = 3e10 * 10
rate = 0.5 
years = 23

slip = rate * years

moment = length * depth * shear_modulus * slip
print(f"{moment=}")

# moment = 15e25
magnitude = moment_to_moment_magnitude(moment)

print(f"{moment=}")
print(f"{magnitude=}")
