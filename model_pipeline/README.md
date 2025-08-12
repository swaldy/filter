# Instructions
efficiency.ipynb contains the code to produce the signal efficiency plot as a function of true transverse-momentum. The default python 3.11.11 venv is used to run the notebook. 
One will also require functions from the pretrain-data-prep repository from smartpix-ml and the .gitmodules should take care of this dependency.
```
git clone --recursive https://github.com/smart-pix/filter.git
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull
```