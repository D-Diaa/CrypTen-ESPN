
cd /tmp/pycharm_project_383 || exit
conda activate pillar_espn
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=7
python3 scripts/accuracy_experiments.py
