# Pay It Forward: A Sustainable, Fair, and Personalized Data Market via Federated Learning

This is the implementation of the paper "Pay It Forward: A Sustainable, Fair, and Personalized Data Market via Federated Learning".

## Set up
- Create a virtual environment
- Install the required packages by running the following command:
```
pip install -r requirements.txt
```
- prepare datasets by running the following command, take Cifar-10 as an example:
```
cd ./dataset/Generator
# python generate_Cifar10.py noniid - pat # for pathological noniid and unbalanced scenario
python generate_Cifar10.py noniid - dir # for practical noniid and unbalanced scenario
```

## Architecture
- Homo encryption: CKSS encryption implemented using tenseal
- Dataset: MNIST, Cifar10, Cifar100, AG news
- Models: CNN, textCNN, DNN, Resnet

## Simulation
- Arguments:
```
'-n_clients' to indicate NUMBER OF CLIENTS
'-model' to indicate MODEL 
'-dataset' to indicate DATASET
'-attack_type' to indicate ATTACK TYPE
'-attack_ratio'to indicate ATTACK DENSITY
'-round' to indicate TRAINING ROUND
'-walk' to indicate WALKING STRATEGY
'-agg' to indicate AGGREGATION METHOD
```
- Run the simulation:
```
python main.py -n_clients 20  -model CNN -dataset Cifar10 -round 200 -walk greedy 
```

## TODO List
- Add training process for different models and dataset
- Add multiple attack scenarios
