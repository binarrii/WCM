# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("1 ==> before import")
import tensorflow as tf
print("2 ==> after import")

print(f"3 ==> {tf.__version__}")
print(f"4 ==> {tf.test.is_built_with_cuda()}")
print(f"5 ==> {tf.sysconfig.get_build_info()}")
print(f"6 ==> {tf.config.list_physical_devices()}")
print(f"7 ==> {tf.config.list_physical_devices('CPU')}")
print(f"8 ==> {tf.config.list_physical_devices('GPU')}")
print("9 ==> done")
