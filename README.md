# hytunaflow

## examples
- see [examples](./examples/)
### train example
simple train
```sh
python train.py
```
change params
```sh
python train.py train.params.p1=-1
```

train with specified config
```sh
python train.py train=t2
```

train with specified config and chagen params
```sh
python train.py train=t2 train.params.p2=100
```

### optuna example
simple tune
```sh
python hypara_tune.py
```
change params
```sh
python hypara_tune.py optuna.params.n_trials=5
```

change train params
```sh
python hypara_tune.py train.params.p1=0
```

tune with specified config
```sh
python hypara_tune.py optuna=op2
```

restart tune
```sh
python hypara_tune.py optuna=op1 +optuna.params.restart_expname={exp_name} +optuna.params.restart_runname={run_name}
```