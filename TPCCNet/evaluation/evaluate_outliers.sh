DATASET_PATH='testdata_various_outliers.h5'
CASE='various_outliers'

# Just some colors to lines on the terminal.
RED=`tput setaf 1`
ORANGE=`tput setaf 3`
DEFAULT=`tput sgr0`

################### Create dataset to test the networks ###################
if [ -f $DATASET_PATH ]; then
	echo "${RED}Testing dataset already exists! Ensure it doesnt affect the testing!${DEFAULT}"
else
	echo "${RED}Creating test dataset ...{DEFAULT}"
	python3 create_statistical_data.py --case $CASE --name $DATASET_PATH
fi
echo -e ''

################### Tests of tpcc-PointNetLK, tpcc-ICP, tpcc-DCP ###################
tpccNET='True'
echo -e "${RED}Using tpccNet: "$tpccNET"\n${DEFAULT}"
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	RESULTS_DIR='results_tpccnet_'$reg_algorithm'_various_outliers'
	echo -e "${ORANGE}Registration Algorithm Getting Tested: "$REG_ALGO"\n${DEFAULT}"

	# Test the network for different angles from 0 to 90 degrees.
	for group_no in $(seq 0 10 90) + $(seq 100 100 1000)
	do
		GROUP_NAME='outliers_'$group_no
		echo "${ORANGE}Value of Initial Misalignment: "$group_no"${DEFAULT}"
		python3 evaluate_stats.py --dataset_path $DATASET_PATH --results_dir $RESULTS_DIR --group_name $GROUP_NAME --tpccnet $tpccNET --reg_algorithm $REG_ALGO
	done
done

################### Tests of PointNetLK, ICP, DCP ###################
tpccNET='False'
echo -e "${RED}Using tpccNet: "$tpccNET"\n${DEFAULT}"
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	RESULTS_DIR='results_'$reg_algorithm'_various_outliers'
	echo -e "${ORANGE}Registration Algorithm Getting Tested: "$REG_ALGO"\n${DEFAULT}"

	# Test the network for different angles from 0 to 90 degrees.
	for group_no in $(seq 0 10 90) + $(seq 100 100 1000)
	do
		GROUP_NAME='outliers_'$group_no
		echo "${ORANGE}Value of Initial Misalignment: "$group_no"${DEFAULT}"
		python3 evaluate_stats.py --dataset_path $DATASET_PATH --results_dir $RESULTS_DIR --group_name $GROUP_NAME --tpccnet $tpccNET --reg_algorithm $REG_ALGO
	done
done


################### Create Plots ###################
CASE='outlier'
for REG_ALGO in 'pointnetlk' 'icp' 'dcp'
do
	for METRIC in 'rotation_error' 'translation_error'
	do
		python3 plot_results.py --case $CASE --reg_algorithm $REG_ALGO --metric $METRIC
	done
done