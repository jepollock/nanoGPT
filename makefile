# Make the various interim files.

.PHONY: task1_example task2_temperature clean task3_log_prob task4_forced_response


task1_example:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=3 --show_probs=True --image_filename=task1_example --start="I live in" --enable_debug=True


task2_temperature:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=1 --show_probs=True --image_filename=task2_temp_0_01 --temperature=0.01 --start="I live in"
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=1 --show_probs=True --image_filename=task2_temp_1_0 --temperature=1.0 --start="I live in"
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=1 --show_probs=True --image_filename=task2_temp_10_0 --temperature=10.0 --start="I live in"

task3_log_prob:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=5  --temperature=0.0001 --start="I live in" --show_total_probability=True | tail -3 > task3_log_prob.txt


task4_fixed_response:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --start="I live in" --show_total_probability=True --fixed_response=" the" | tail -3 > task4_fixed_response.txt
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --start="I live in" --show_total_probability=True --fixed_response=" the capital"| tail -3 >> task4_fixed_response.txt
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --start="I live in" --show_total_probability=True --fixed_response=" the capital of"| tail -3 >> task4_fixed_response.txt
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --start="I live in" --show_total_probability=True --fixed_response=" the capital of New Zealand" | tail -3 >> task4_fixed_response.txt


task5_dataset:
	python convert.py --split=False --max_rows=10 --in_file=data/gms8k/test-00000-of-00001.parquet --train_file=test.json
	python convert.py --in_file=data/gms8k/train-00000-of-00001.parquet

task6_prepare_dataset:
	python data/gms8k/prepare.py


clean:
	rm task1_example*.png
	rm task2_temp*.png
	rm task3_log_prob.txt
	rm task4_*response.txt
	rm test.json
	rm data/gms8k/*.json
