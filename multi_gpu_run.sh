echo "Start"

set -euo pipefail

main_path='/home/Eq_Solver/'
BS=32
num_layers_gnn=5
chan_gnn=64

lr=1e-3

epochs=1e3

python -u Eq_GNN_Saving.py --arg_main_path ${main_path} > pyouts_saving.out

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 Eq_GNN_Multi.py --arg_main_path ${main_path} --arg_BS ${BS} --arg_num_layers_gnn ${num_layers_gnn} --arg_chan_gnn ${chan_gnn} --arg_epochs ${epochs} --arg_lr ${lr} > pyouts.out