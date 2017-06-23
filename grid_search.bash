#!/bin/bash

for ((lr=40; lr <= 40; lr+=1))
do
    for ((bz=256; bz <= 256; bz+=10))
    do
	mkdir param-test-${lr}-${bz}
	cd param-test-${lr}-${bz}
	cp ../cnn_models.py .
	cp ../utils.py .
	cp ../train_cnn.py .
	sed -i "s/BATCH_SIZE = to_be_replaced/BATCH_SIZE = ${bz}/g" train_cnn.py
	sed -i "s/LR = to_be_replaced/LR = ${lr}/g" train_cnn.py

	cat > auto_${lr}_${bz}.pbs << FIN
#!/bin/bash
#SBATCH --job-name=cifar10
#SBATCH --time=5:0:0
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=60000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xzhang11@stanford.edu

python train_cnn.py

wait
FIN
	qsub auto_${lr}_${bz}.pbs
        #python train_cnn.py
	cd ..
    done
done

