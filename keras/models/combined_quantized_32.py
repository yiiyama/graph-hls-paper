import models.combined_32 as mod

initial_lr = mod.initial_lr
generator_args = mod.generator_args
compile_args = mod.compile_args
root_out_dtype = mod.root_out_dtype

def make_model():
    return mod._make_model(True)

make_loss = mod.make_loss
evaluate_prediction = mod.evaluate_prediction
make_root_out_entries = mod.make_root_out_entries
make_h5_out_data = mod.make_h5_out_data
write_ascii_out = mod.write_ascii_out
