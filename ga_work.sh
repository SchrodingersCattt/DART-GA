export exp_name=FeNiCoCrVTi_V0.2_Cr0.1_AlTi0.1
export element="Fe,Ni,Co,Cr,V,Al,Ti"  
export init_mode='init'
export population_size=30
export constraints="V<0.2,Cr<0.1,(Al+Ti)<0.1"
export selection_mode='roulette'
export get_density_mode='weighted_avg'
export output=$(date +'%Y%m%d_%H%M%S')_${init_mode}_${exp_name}_ga-${selection_mode}_fcc.log
. /mnt/data_nas/public/.bashrc
conda activate prop_pred

cd /mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/Genetic_Alloy

python ga.py -o ${output} --elements ${element} --init_mode ${init_mode}  --a 0.8 --b 0.1 --c 0.1 --d 0.0 --constraints ${constraints} --selection_mode ${selection_mode} --get_density_mode ${get_density_mode} --population_size ${population_size}