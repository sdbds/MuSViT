#!/bin/bash

models=("PRAIG/musvit" "PRAIG/musvit-light")
datasets=("cipi" "fs" "ps")
architectures=("rnn" "transformer" "mlp")
start=0
end=4
timestamp=$(date +"%Y-%m-%d-%H:%M:%S")

for architecture in "${architectures[@]}"; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            jobid=""
            for ((i=start; i<=end; i++)); do
                safe_model="${model//\//_}"
                job_file="job_${safe_model}_${dataset}_${architecture}_${i}.slurm"
                sed "s|{MODEL}|${model}|g; s|{DATASET}|${dataset}|g; s|{ARCHITECTURE}|${architecture}|g; s|{I}|${i}|g; s|{END}|${end}|g; s|{TIMESTAMP}|${timestamp}|g;" launch_experiment.slurm > "$job_file"

                if [[ -z "$jobid" ]]; then
                    jobid=$(sbatch "$job_file" | awk '{print $4}')
                else
                    jobid=$(sbatch --dependency=afterany:${jobid} "$job_file" | awk '{print $4}')
                fi

                rm "$job_file"
            done
        done
    done
done
