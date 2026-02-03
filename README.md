# Terminal setup
cd smart_pix
source .venv/bin/activate

# Filtering algorithm

Pipeline to run all codes: pre_processing -> neuralNetwork -> data_reduction -> plotting & turnOnCurves.

pre_processing.ipynb reads all datasets from a given location (say cmslpc or lxplus), and shuffles and separates them into training and testing datasets, and saves the corresponding output files locally. These files will then be sent as input to neuralNetwork.ipynb. Note - A set of files is produced for one pT boundary; this file has to be run once for every pT boundary. This was set up in a rather cumbersome way owing to requiring checks on the truth distributions and event numbers at every pT boundary. 
NOTE: pre-processing.py can now run for all pT boundaries at one go and save validation/check information for every pT boundary!

neuralNetwork.ipynb trains and evaluates the model on the test and train datasets, respectively.

data_reduction.ipynb quantifies the model's performance (i.e., calculate signal efficiency, background rejection, and data reduction).

The remaining files help produce the relevant performance results and plots.

## ASIC testing suite
model_pipeline/ contains a set of codes needed for ASIC testing. It also includes efficiency.ipynb to produce plots of signal efficiency as a function of true transverse-momentum. The default python 3.11.11 venv is used to run the notebook. 
One will also require functions from the pretrain-data-prep repository from smartpix-ml and the .gitmodules should take care of this dependency.
```
git clone --recursive https://github.com/smart-pix/filter.git
cd filter
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull
```
After encountering instances where the submodules don't properly install, one can try the following as an alternative to manually add the submodules within the filter directory:
```
git submodule add https://github.com/smart-pix/pretrain-data-prep.git
git submodule update --init --recursive
```
