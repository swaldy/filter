git clone --recursive https://github.com/smart-pix/filter.git
git submodule foreach --recursive 'if git show-ref --verify --quiet refs/heads/main; then git checkout main; elif git show-ref --verify --quiet refs/heads/master; then git checkout master; else echo "Neither main nor master branch exists"; fi'
git submodule foreach --recursive git pull

# git clone https://github.com/smart-pix/pretrain-data-prep.git
# cp pretrain-data-prep/dataset_utils.py ./
