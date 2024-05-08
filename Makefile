main_task_chicken:
	cd build && ./ADPCCSO_optimization --d-coef-mult=0.1 --best=false

main_task_fish:
	cd build && ./ADPCCSO_optimization --swarms=fish --d-coef-mult=0.1 --best=false

rastrigin_chicken_2_dim:
	cd build && ./ADPCCSO_optimization --fitness=rastrigin --number-starts=16 --multistart-threads=8

rastrigin_fish_2_dim:
	cd build && ./ADPCCSO_optimization --swarms=fish --fitness=rastrigin --number-starts=16 --multistart-threads=8

rastrigin_chicken_high_dim:
	cd build && ./ADPCCSO_optimization --fitness=rastrigin --number-starts=16 --multistart-threads=8 --dim=500 --num-agents=500

help:
	cd build && ./ADPCCSO_optimization -h

