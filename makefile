# Make the various interim files.

.PHONY: task1_example task2_temperature clean task3_log_prob task4_forced_response


task1_example:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=3 --show_probs=True --image_filename=task1_example --start="I live in" 


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

task7_finetune:
	python train.py config/finetune_gms8k.py --device=mps

task8_eval_baseline:
	python -u ./eval.py --top_k=50000 --show_total_probability=True --num_samples=1 --init_from=gpt2-large --device=mps --eval_data_file=test.json --use_eval_response=True | tee task8_baseline_fixed_response.txt
	python -u ./eval.py --show_total_probability=True --num_samples=1 --init_from=gpt2-large --device=mps --eval_data_file=test.json --use_eval_response=False | tee task8_baseline_unfixed_response.txt

task8_eval_finetune:
	python -u ./eval.py --top_k=50000 --show_total_probability=True --num_samples=1 --init_from=resume --out_dir=out-gms8k --device=mps --eval_data_file=test.json --enable_response_start_token=True --enable_stop_token=True --use_eval_response=True | tee task8_finetune_fixed_response.txt
	python -u ./eval.py --show_total_probability=True --num_samples=1 --init_from=resume --out_dir=out-gms8k --device=mps --eval_data_file=test.json --enable_response_start_token=True --enable_stop_token=True --use_eval_response=False | tee task8_finetune_unfixed_response.txt


task9_beam_search:
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=3 --start="I live in" --show_total_probability=True | tee task9_beam_search_size1.txt
	python sample.py --init_from=gpt2-large --num_samples=1 --device=mps --max_new_tokens=3 --start="I live in" --beam_width=5 --show_total_probability=True | tee task9_beam_search_size5.txt


clean:
	rm task1_example*.png
	rm task2_temp*.png
	rm task3_log_prob.txt
	rm task4_*response.txt
	rm test.json
	rm data/gms8k/*.json
	rm task8_baseline_*fixed_response.txt
	rm task8_finetune_*fixed_response.txt
