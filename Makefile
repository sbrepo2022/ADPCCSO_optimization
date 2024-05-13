build:
	cd build && make

main_task_sphere:
	cd build && ./ADPCCSO_optimization --fitness=main_task_sphere --d-coef-mult=0.1 --best=false

main_task_cube:
	cd build && ./ADPCCSO_optimization --fitness=main_task_cube --d-coef-mult=0.1 --best=false

main_task_simplex:
	cd build && ./ADPCCSO_optimization --fitness=main_task_simplex --d-coef-mult=0.1 --best=false

main_task_fish:
	cd build && ./ADPCCSO_optimization --swarms=fish --d-coef-mult=0.1 --best=false

rastrigin_chicken_2_dim:
	cd build && ./ADPCCSO_optimization --fitness=rastrigin --number-starts=16 --threads=8

rastrigin_fish_2_dim:
	cd build && ./ADPCCSO_optimization --swarms=fish --fitness=rastrigin --number-starts=16 --threads=8

rastrigin_both_2_dim:
	cd build && ./ADPCCSO_optimization --swarms=both --fitness=rastrigin --number-starts=16 --threads=16 --num-agents-for-exchange=10

high_load_chicken:
	cd build && ./ADPCCSO_optimization --fitness=high_load --parallel-strategy=swarm --number-starts=1 --threads=8

rastrigin_chicken_high_dim:
	cd build && ./ADPCCSO_optimization --fitness=rastrigin --number-starts=16 --threads=8 --dim=500 --num-agents=500

3d:
	cd build && unset GTK_PATH && gnuplot plot_script_3d.gp

2d:
	cd build && unset GTK_PATH && gnuplot plot_script_2d.gp

stagnation:
	cd build && unset GTK_PATH && gnuplot plot_script_stagnation.gp

help:
	cd build && ./ADPCCSO_optimization -h

