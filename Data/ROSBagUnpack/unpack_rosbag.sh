#!/usr/bin/env bash

port=11321
bagfile=$1
help_string="Usage:\n./unpack_rosbag.sh [bagfile]"

topic_list=("/whiskeye/head/pose" "/whiskeye/head/cam0/image_raw/compressed")
file_list=("raw_pose.csv" "raw_images_cam0.csv")

subprocess_ids=()
listener_ids=()

if [ -z $1 ]; then

	echo "Please specify the rosbag to unpack."

	echo -e $help_string

	exit 1

fi

if [ ! -e $1 ]; then

	echo "File \"$1\" not found in this directory"

	echo -e $help_string

	exit 1

fi

echo "Starting ROS instance for unpacking on port $port"

export ROS_MASTER_URI=http://localhost:$port

roscore -p $port >/dev/null 2>/dev/null & roscore_id=$!

sleep 5

echo "Starting master process here"
echo "Do NOT close this shell until all topics are confirmed complete!"

for i in ${!topic_list[@]}; do

	echo "Unpacking topic ${topic_list[i]} to ${file_list[i]}"

	rostopic echo -b $bagfile -p ${topic_list[i]} > ${file_list[i]} & subprocess_ids+=("$!")

	echo "Subprocess dispatched on PID ${subprocess_ids[i]}"

	sleep 1

done

function cleanup() {

echo ""

for i in ${!subprocess_ids[@]}; do

	if kill -0 ${subprocess_ids[i]} 2>/dev/null; then

		echo "Killing unfinished process ${subprocess_ids[i]}"

		kill ${subprocess_ids[i]} 2>/dev/null

	fi

done

if kill -0 $roscore_id 2>/dev/null; then

        echo "Killing roscore on port $port"

	kill $roscore_id

fi

exit 0

}

trap cleanup SIGINT

while ! [ ${#subprocess_ids[@]} -eq 0 ]; do

	for i in ${!subprocess_ids[@]}; do

		if ! kill -0 ${subprocess_ids[i]} 2>/dev/null; then

			echo "Process ${subprocess_ids[i]} completed"

			unset subprocess_ids[i]

		fi
	
	done

	if [ ${#subprocess_ids[@]} -eq ${#listener_ids[@]} ]; then

		for i in ${!listener_ids[@]}; do

			echo "No more input for listener ${listener_ids[i]}; ending process"

			kill -2 ${listener_ids[i]} >/dev/null 2>/dev/null

			unset subprocess_ids[i] && unset listener_ids[i]

		done
	fi

done

echo "All files written, killing roscore on port $port"

if kill -0 $roscore_id 2>/dev/null; then

        kill $roscore_id 2>/dev/null

fi

echo "Calling image extraction script"

for i in ${!topic_list[@]}; do
	
	if [[ ${topic_list[i]} =~ camera ]]; then

		mkdir -p ${BASH_REMATCH[0]}

	fi

	if [[ ${topic_list[i]} =~ cam[0-9] ]]; then

		mkdir -p ${BASH_REMATCH[0]}

	fi
done

python3 extract_images_from_rosbag.py $bagfile &

wait

echo "Calling postprocessing script"

python3 unpack_rosbag.py &

wait

echo "Zipping all generated files and images to rosbag_output.zip"

zip rosbag_output.zip cam0/*.jpg cam1/*.jpg *.csv *.npy >/dev/null




