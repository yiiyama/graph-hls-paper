GarNet on HLS4ML
================
Dataset and code for [https://arxiv.org/abs/2008.03601](https://arxiv.org/abs/2008.03601)

## Simulation sample generation

Calorimeter simulation was forked from [jkiesele/miniCalo](https://github.com/jkiesele/miniCalo) although there is little resemblance at this point. To compile and run,
```
cd generation
source env.sh
make generate

./generate -g -f geom.root   # Will be used later in the data preprocessing step
./run_generate.sh PART EMIN EMAX NGEN JOBID SEEDBASE
```
Where the last command should be run in a batch system with JOBID set to an integer single-job identifier (used to set random seeds by SEED=SEEDBASE+JOBID). PART is either electron, pioncharged, or pileup. For the studies in the paper, EMIN=10 and EMAX=100 were used.

Generated event files need to be preprocessed (randomly mix electron and pion events, overlay pileup, form clusters and write the data hit-wise (instead of sensor-wise)):
```
# In an environment where numpy, ROOT, h5py, and uproot are available:
cd generation
./mixing.py --dataset=combined --format=root-sparse --nevt=10000 --nfile=100 --source=output_directory_of_generate --out=directory_and_filename_base --add-pu
```
Option `--dataset=combined` sets up the dataset for combined regression+classification task. Output format `root-sparse` is easier to use for HEPists and makes the training faster because the entire dataset (up to certain size of course) fits in RAM. Published dataset in [https://zenodo.org/record/3888910#.X6yQXJMzbUI](https://zenodo.org/record/3888910#.X6yQXJMzbUI) was preprocessed with `--dataset=public` and `--format=h5`.

## Training and weights output

Everything keras is organized under the `keras` directory. Keras layers are defined in `layers` (includes a git submodule caloGraphNN), models in `models`, and data generators in `generators`. Usage is
```
./train.py combined --train training_data_files --validate validation_data_files --out weights_output_file --batch-size 64 --num-epochs 1000 --input-type root-sparse
```
The first argument is the model name, which corresponds to the file name in the `models` directory. The model `combined` corresponds to the "continuous Nmax=128" model in the paper. The other models in the paper are available as `combined[_quantized][_N].py`. Use the output weight file to run tests:
```
./predict.py combined weights_file test_data_file --input-type root-sparse --ascii-out tb_output_predictions.dat
```
The last `--ascii-out` is used to compare the predictions with the HLS C simulation later.

## HLS

You need a Vivado license to run this part. Have hls4ml installed in your environment. GarNet is already in the official repository, so you can create a project simply by
```
cd hls4ml
hls4ml convert -c config_prequantization.yml
```
By default, the yml configuration uses the weight file already in the `hls4ml` directory. Replace the file with yours or edit the configuration to point to your file.