from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

print("=========All Variables==========")
print_tensors_in_checkpoint_file("tmp/model.ckpt", tensor_name=None, all_tensors=True, all_tensor_names=True)